//+------------------------------------------------------------------+
//|                                                     SavgolEA.mq5 |
//| Expert Advisor "Savgol" - live port of TradeSavgol.py, built on  |
//| two custom indicators (must be compiled/available) - only one is |
//| loaded, matching whichever mode InpSignalMode selects:            |
//|   mode 1 (SIGNAL_CONFLUENCE)   -> "Savgol"      (Savgol.mq5)      |
//|   mode 2 (SIGNAL_SAVGOL_PIVOT) -> "SavgolPrice" (SavgolPrice.mq5) |
//| Deployed as "Savgol_EA" to avoid a name collision with the        |
//| "Savgol" indicator (the Strategy Tester bundles iCustom            |
//| dependencies by name, and two same-named artifacts confused it).  |
//|                                                                  |
//| InpSignalMode selects which of two independent entry/exit rules   |
//| drives the EA. Only one is active at a time.                      |
//|                                                                  |
//| --- Mode 1: SIGNAL_CONFLUENCE (default, unchanged) ---------------|
//| Entry signal, reproducing find_entry_signals() exactly: a raw    |
//| MA10 pivot and a Savgol(close) pivot (indicator buffers 3-6)     |
//| within InpMaxGapBars of each other ->                            |
//|   MA10 valley + Savgol(close) valley -> LONG                     |
//|   MA10 peak   + Savgol(close) peak   -> SHORT                    |
//| taken InpEntryDelayBars after the later of the two pivots, at    |
//| market (the live analog of "that bar's open" in the backtest).  |
//| Same caveat as mode 2 below applies here too, to a lesser degree: |
//| Savgol(close) pivots within the last SG_HALF (7) bars are built   |
//| from still-revising provisional values, so a very recent pivot    |
//| could in principle be revised away. Requiring BOTH an MA10 pivot  |
//| AND a Savgol(close) pivot to agree (the confluence rule) makes    |
//| this much less likely to matter in practice than in mode 2's bare |
//| single-series pivot check, but it is not immune to it.            |
//|                                                                  |
//| Exit, reproducing simulate_trades(): stop loss / take profit at  |
//| fixed InpStopLossPct / InpTakeProfitPct (set as native SL/TP on  |
//| the order, so the broker enforces them - no polling needed), OR  |
//| an opposite-side entry signal, whichever comes first.            |
//|                                                                  |
//| Deliberate difference from the Python backtest: TradeSavgol.py's |
//| simulate_trades() evaluates every entry as an independent         |
//| hypothetical trade, so signals can overlap in time. That is an  |
//| analysis convenience, not a tradable portfolio. This EA instead  |
//| runs ONE position at a time (works on both netting and hedging   |
//| accounts): a same-side signal while already in that side is      |
//| ignored (already in the trade), an opposite-side signal closes    |
//| the open position and opens the new one - matching the backtest's |
//| "opposite_signal" exit exactly, just without the overlapping      |
//| hypothetical trades.                                              |
//|                                                                  |
//| --- Mode 2: SIGNAL_SAVGOL_PIVOT (new) ------------------------------|
//| Built on a separate, slimmed-down indicator, "SavgolPrice"          |
//| (SavgolPrice.mq5) - one Savitzky-Golay smoothed price series and   |
//| its peaks/valleys, no MA10, no confluence. InpPriceSource selects  |
//| which price feeds the filter: Close, Open, High, Low, or OHLC/4.  |
//|   - PEAK   -> SHORT opportunity (close long first, if one is      |
//|     open; open short if none is open yet).                        |
//|   - VALLEY -> LONG opportunity (close short first, if one is      |
//|     open; open long if none is open yet).                         |
//| These entries have no stop loss or take profit, for now.          |
//| A pivot is only acted on once it is SG_HALF (7) bars old, read at   |
//| shift SG_HALF for the same reason mode 1's confluence check waits: |
//| the smoothed line is a centered filter, so the indicator's last    |
//| SG_HALF bars are provisional edge estimates that keep changing as  |
//| new bars arrive (and a "peak"/"valley" flagged in that window can  |
//| be revised away once the value it was based on changes), only      |
//| becoming final SG_HALF bars later. The chart's plotted arrows       |
//| still show pivots in that live tail for responsiveness - only the  |
//| trading decision waits for the pivot's value to settle, which      |
//| costs ~7-8 bars of lag between the arrow appearing on the chart    |
//| and the ticket actually opening.                                   |
//|                                                                  |
//| Only signals dated after the EA was attached are acted on (no    |
//| backlog of historical signals fires on startup) - applies to      |
//| both modes.                                                       |
//|                                                                  |
//| --- Position monitoring after an open ticket (both modes) --------|
//| No local "am I in a trade" flag is kept - CurrentPositionSide()   |
//| re-derives it from the broker's own position list every time, so  |
//| it self-heals regardless of how a position actually closed (SL,   |
//| TP, manual close, this EA). This account is hedging mode, where   |
//| more than one position per symbol can exist at once, so plain     |
//| PositionSelect(symbol)/CTrade::PositionClose(symbol) are NOT      |
//| reliable (the latter only closes the first match it finds) -      |
//| every position is enumerated and closed by ticket instead. A      |
//| close/open trade request's success is checked; on failure the      |
//| signal is left unmarked so it retries next bar rather than being   |
//| silently dropped (which could otherwise leave both a stale         |
//| position and a new opposite one open at the same time).           |
//|                                                                  |
//| --- Close-then-reverse never happens on the same bar (both modes) -|
//| When a signal is opposite to the currently open ticket, that       |
//| ticket is closed immediately, but the reverse entry is queued       |
//| (see TryFulfillPendingEntry()/QueuePendingEntry()) and only opened  |
//| on the NEXT new bar - one full bar flat in between, never close    |
//| and open on the same bar. Fresh signal detection is skipped        |
//| entirely on a bar that fulfills a pending entry.                  |
//+------------------------------------------------------------------+
#property copyright "Savgol"
#property version   "3.00"

#include <Trade\Trade.mqh>

enum ENUM_SIGNAL_MODE
{
   SIGNAL_CONFLUENCE   = 0, // MA10 pivot + Savgol(close) pivot confluence (mode 1, original)
   SIGNAL_SAVGOL_PIVOT = 1  // SavgolPrice peak/valley alone (mode 2, new)
};

// Must match SavgolPrice.mq5's ENUM_SAVGOL_PRICE_SOURCE order exactly - passed
// positionally to that indicator via iCustom() in OnInit().
enum ENUM_SAVGOL_PRICE_SOURCE
{
   SAVGOL_PRICE_CLOSE = 0, // Close
   SAVGOL_PRICE_OPEN  = 1, // Open
   SAVGOL_PRICE_HIGH  = 2, // High
   SAVGOL_PRICE_LOW   = 3, // Low
   SAVGOL_PRICE_OHLC4 = 4  // (Open+High+Low+Close)/4
};

input ENUM_SIGNAL_MODE InpSignalMode = SIGNAL_CONFLUENCE; // which entry/exit rule drives the EA
input int    InpMAPeriod      = 10;    // (mode 1) MA period (must match the Savgol indicator's InpMAPeriod)
input int    InpMaxGapBars    = 3;     // (mode 1) max bars between MA10 pivot and Savgol(close) pivot to count as confluence
input int    InpEntryDelayBars= 2;     // (mode 1) bars after the later pivot before entry (matches TradeSavgol.py)
input double InpStopLossPct   = 5.0;   // (mode 1) stop loss, % of entry price
input double InpTakeProfitPct = 30.0;  // (mode 1) take profit, % of entry price
input int    InpLookbackBars  = 300;   // (mode 1) history window rescanned each bar for signals
input ENUM_SAVGOL_PRICE_SOURCE InpPriceSource = SAVGOL_PRICE_CLOSE; // (mode 2) price fed into the SavgolPrice indicator
input double InpLots          = 0.10;  // order volume
input ulong  InpMagic         = 20260928;

CTrade trade;
int    g_handle      = INVALID_HANDLE; // "Savgol" indicator (mode 1)
int    g_priceHandle = INVALID_HANDLE; // "SavgolPrice" indicator (mode 2)
datetime g_lastBarTime = 0;
datetime g_lastExecutedEntryTime = 0;

// A close-then-reverse never opens the new side on the same bar it closed the
// old one - the reverse entry is queued here and only opened on the NEXT new
// bar (see TryFulfillPendingEntry()).
bool     g_pendingActive     = false;
int      g_pendingSide       = -1;
datetime g_pendingSignalTime = 0;
bool     g_pendingUseStops   = false;

// -- "Savgol" indicator buffer layout (mode 1) --
#define BUF_MA10           0
#define BUF_SAVGOL_CLOSE   1
#define BUF_SAVGOL_MA10    2
#define BUF_MA10_PEAK      3
#define BUF_MA10_VALLEY    4
#define BUF_SAVGOL_C_PEAK  5
#define BUF_SAVGOL_C_VALLEY 6

// -- "SavgolPrice" indicator buffer layout (mode 2) --
#define PRICE_BUF_LINE     0
#define PRICE_BUF_PEAK     1
#define PRICE_BUF_VALLEY   2

#define SG_HALF            7   // must match both indicators' centered-filter half-window (SG_WINDOW=15)

//+------------------------------------------------------------------+
void ReverseArray(const double &src[], double &dst[], int n)
{
   ArrayResize(dst, n);
   for(int i = 0; i < n; i++)
      dst[i] = src[n - 1 - i];
}

void ReverseTimeArray(const datetime &src[], datetime &dst[], int n)
{
   ArrayResize(dst, n);
   for(int i = 0; i < n; i++)
      dst[i] = src[n - 1 - i];
}

//+------------------------------------------------------------------+
//| Chronological (oldest-first) indices where buf[i] != EMPTY_VALUE  |
//+------------------------------------------------------------------+
void CollectPivotPositions(const double &buf[], int n, int &positions[])
{
   ArrayResize(positions, 0);
   for(int i = 0; i < n; i++)
   {
      if(buf[i] != EMPTY_VALUE)
      {
         int sz = ArraySize(positions);
         ArrayResize(positions, sz + 1);
         positions[sz] = i;
      }
   }
}

//+------------------------------------------------------------------+
//| Direct port of TradeSavgol.py's find_entry_signals() greedy       |
//| confluence matching for one side (LONG uses valleys, SHORT uses  |
//| peaks on both series).                                            |
//+------------------------------------------------------------------+
void MatchSide(const int &maPositions[], const int &sigPositions[], int maxGap, int entryDelay, int n,
               int &outEntryPos[], int side, int &outSide[])
{
   int nSig = ArraySize(sigPositions);
   bool used[];
   ArrayResize(used, nSig);
   ArrayInitialize(used, false);

   int nMa = ArraySize(maPositions);
   for(int a = 0; a < nMa; a++)
   {
      int maPos = maPositions[a];
      int bestIdx = -1;
      int bestDist = INT_MAX;
      for(int b = 0; b < nSig; b++)
      {
         if(used[b]) continue;
         int dist = MathAbs(sigPositions[b] - maPos);
         if(dist <= maxGap && dist < bestDist)
         {
            bestDist = dist;
            bestIdx  = b;
         }
      }
      if(bestIdx < 0) continue;
      used[bestIdx] = true;

      int pivotPos    = MathMax(maPos, sigPositions[bestIdx]);
      int entryPos    = pivotPos + entryDelay;
      if(entryPos >= n) continue; // entry bar not available in this window yet

      int sz = ArraySize(outEntryPos);
      ArrayResize(outEntryPos, sz + 1);
      ArrayResize(outSide,     sz + 1);
      outEntryPos[sz] = entryPos;
      outSide[sz]     = side;
   }
}

//+------------------------------------------------------------------+
int OnInit()
{
   // iCustom() only computes the indicator's buffers for this EA to read - it
   // does NOT draw anything on the chart by itself. Attach it to the main
   // chart window explicitly so its lines/arrows are visible. Only the
   // indicator the active mode actually needs is loaded.
   if(InpSignalMode == SIGNAL_CONFLUENCE)
   {
      g_handle = iCustom(_Symbol, PERIOD_CURRENT, "Savgol", InpMAPeriod);
      if(g_handle == INVALID_HANDLE)
      {
         Print("Savgol EA: failed to load Savgol indicator (iCustom). Make sure Savgol.ex5 is compiled and in MQL5\\Indicators.");
         return(INIT_FAILED);
      }
      if(!ChartIndicatorAdd(0, 0, g_handle))
         Print("Savgol EA: could not draw the Savgol indicator on the chart (error ", GetLastError(), ") - EA will still trade normally.");
   }
   else // SIGNAL_SAVGOL_PIVOT
   {
      g_priceHandle = iCustom(_Symbol, PERIOD_CURRENT, "SavgolPrice", InpPriceSource);
      if(g_priceHandle == INVALID_HANDLE)
      {
         Print("Savgol EA: failed to load SavgolPrice indicator (iCustom). Make sure SavgolPrice.ex5 is compiled and in MQL5\\Indicators.");
         return(INIT_FAILED);
      }
      if(!ChartIndicatorAdd(0, 0, g_priceHandle))
         Print("Savgol EA: could not draw the SavgolPrice indicator on the chart (error ", GetLastError(), ") - EA will still trade normally.");
   }

   trade.SetExpertMagicNumber(InpMagic);

   // Only act on signals from after the EA was attached - no backlog on startup.
   g_lastExecutedEntryTime = TimeCurrent();

   Print("Savgol EA: active mode = ", EnumToString(InpSignalMode));

   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   if(g_handle != INVALID_HANDLE)
      IndicatorRelease(g_handle);
   if(g_priceHandle != INVALID_HANDLE)
      IndicatorRelease(g_priceHandle);
}

//+------------------------------------------------------------------+
bool IsNewBar()
{
   datetime t = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(t != g_lastBarTime)
   {
      g_lastBarTime = t;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
bool OpenPosition(int side, datetime entryTime)
{
   double price = (side == 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double sl, tp;
   string comment = StringFormat("Savgol %s %s", side == 0 ? "LONG" : "SHORT", TimeToString(entryTime));

   bool ok;
   if(side == 0) // LONG
   {
      sl = price * (1.0 - InpStopLossPct / 100.0);
      tp = price * (1.0 + InpTakeProfitPct / 100.0);
      ok = trade.Buy(InpLots, _Symbol, price, sl, tp, comment);
   }
   else // SHORT
   {
      sl = price * (1.0 + InpStopLossPct / 100.0);
      tp = price * (1.0 - InpTakeProfitPct / 100.0);
      ok = trade.Sell(InpLots, _Symbol, price, sl, tp, comment);
   }

   if(!ok)
      Print("Savgol EA: failed to open ", side == 0 ? "LONG" : "SHORT", " position - retcode=",
            trade.ResultRetcode(), " (", trade.ResultRetcodeDescription(), ")");
   return ok;
}

//+------------------------------------------------------------------+
//| -1 = flat, 0 = LONG open, 1 = SHORT open (this EA's own magic     |
//| number + symbol only).                                            |
//|                                                                    |
//| This account is confirmed hedging mode, where more than one        |
//| position per symbol can exist simultaneously - plain               |
//| PositionSelect(symbol) is not reliable there (it isn't guaranteed  |
//| to find a specific one), so every open position is enumerated by   |
//| ticket instead. If both a LONG and a SHORT are somehow open at     |
//| once for this symbol/magic (only possible if a close silently      |
//| failed on a previous cycle), that is a bad state: close everything |
//| and report flat, so the EA re-syncs cleanly rather than compound   |
//| the mistake.                                                       |
//+------------------------------------------------------------------+
int CurrentPositionSide()
{
   int longCount = 0, shortCount = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;

      if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
         longCount++;
      else
         shortCount++;
   }

   if(longCount > 0 && shortCount > 0)
   {
      Print("Savgol EA: WARNING - ", longCount, " long and ", shortCount, " short position(s) "
            "open at once for this symbol/magic (should never both be open at the same time) - "
            "closing everything to resync.");
      CloseAllOwnPositions();
      return -1;
   }
   if(longCount > 0) return 0;
   if(shortCount > 0) return 1;
   return -1;
}

//+------------------------------------------------------------------+
//| Closes every open position for this symbol+magic, by ticket -     |
//| CTrade::PositionClose(symbol) only closes the FIRST matching       |
//| position it finds on a hedging account, which would silently      |
//| strand any others. Returns true only if every close succeeded.    |
//+------------------------------------------------------------------+
bool CloseAllOwnPositions()
{
   bool allOk = true;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagic) continue;

      if(!trade.PositionClose(ticket))
      {
         Print("Savgol EA: failed to close position #", ticket, " - retcode=",
               trade.ResultRetcode(), " (", trade.ResultRetcodeDescription(), ")");
         allOk = false;
      }
   }
   return allOk;
}

//+------------------------------------------------------------------+
//| Mode 2 entry: same open action as OpenPosition(), but with no     |
//| stop loss / take profit set.                                     |
//+------------------------------------------------------------------+
bool OpenPositionNoStops(int side, datetime signalTime)
{
   double price = (side == 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   string comment = StringFormat("Savgol XO %s %s", side == 0 ? "LONG" : "SHORT", TimeToString(signalTime));

   bool ok = (side == 0) ? trade.Buy(InpLots, _Symbol, price, 0.0, 0.0, comment)
                          : trade.Sell(InpLots, _Symbol, price, 0.0, 0.0, comment);

   if(!ok)
      Print("Savgol EA: failed to open ", side == 0 ? "LONG" : "SHORT", " position - retcode=",
            trade.ResultRetcode(), " (", trade.ResultRetcodeDescription(), ")");
   return ok;
}

//+------------------------------------------------------------------+
//| Called once per new bar, before any fresh signal detection. If a  |
//| close queued a reverse entry on a previous bar, this is the bar   |
//| that actually opens it - enforcing a full bar of flat time        |
//| between closing a ticket and opening the opposite one (never both |
//| on the same bar). Returns true if a pending entry existed this    |
//| bar (whether or not the open itself succeeded) - the caller       |
//| should then skip fresh signal detection for this bar.             |
//+------------------------------------------------------------------+
bool TryFulfillPendingEntry()
{
   if(!g_pendingActive)
      return false;

   int current = CurrentPositionSide();
   if(current != -1)
   {
      // Shouldn't normally happen (everything was closed right before this was
      // queued), but don't let a stale pending entry block things forever.
      Print("Savgol EA: dropping pending entry - a position is already open.");
      g_pendingActive = false;
      return false;
   }

   bool ok = g_pendingUseStops ? OpenPosition(g_pendingSide, g_pendingSignalTime)
                                : OpenPositionNoStops(g_pendingSide, g_pendingSignalTime);
   if(ok)
      g_pendingActive = false;
   // else: leave it active and retry next bar

   return true;
}

//+------------------------------------------------------------------+
//| Queues a reverse-side entry to be opened on the NEXT bar by        |
//| TryFulfillPendingEntry() - never on the same bar as the close.    |
//+------------------------------------------------------------------+
void QueuePendingEntry(int side, datetime signalTime, bool useStops)
{
   g_pendingActive     = true;
   g_pendingSide        = side;
   g_pendingSignalTime  = signalTime;
   g_pendingUseStops    = useStops;
}

//+------------------------------------------------------------------+
//| Returns true only if the position book ends up correctly handled  |
//| (opened from flat, already in the right side, or the opposite     |
//| side was closed and the reverse queued) - false means the caller  |
//| should NOT treat this signal as handled, so it gets retried on    |
//| the next bar instead of being silently dropped.                   |
//+------------------------------------------------------------------+
bool ProcessNewEntry(int side, datetime entryTime)
{
   int current = CurrentPositionSide();

   if(current == side)
      return true; // already in this direction, nothing to do

   if(current == -1)
      return OpenPosition(side, entryTime); // flat, no ticket to close - open right away

   // Opposite-side signal closes the open position, exactly like
   // simulate_trades()'s "opposite_signal" exit - but the reverse entry
   // itself waits for the next bar (see QueuePendingEntry()).
   if(!CloseAllOwnPositions())
      return false; // don't queue a reverse entry on top of a stuck position

   QueuePendingEntry(side, entryTime, true);
   return true;
}

bool ProcessNoStopsEntry(int side, datetime signalTime)
{
   int current = CurrentPositionSide();

   if(current == side)
      return true; // already in this direction

   if(current == -1)
      return OpenPositionNoStops(side, signalTime);

   if(!CloseAllOwnPositions())
      return false;

   QueuePendingEntry(side, signalTime, false);
   return true;
}

//+------------------------------------------------------------------+
double GetIndicatorValue(int handle, int bufferIndex, int shift)
{
   double arr[];
   ArraySetAsSeries(arr, true);
   if(CopyBuffer(handle, bufferIndex, shift, 1, arr) <= 0)
      return EMPTY_VALUE;
   return arr[0];
}

//+------------------------------------------------------------------+
//| Mode 2 (SIGNAL_SAVGOL_PIVOT): fires on a peak or valley of the     |
//| "SavgolPrice" indicator's single smoothed series (InpPriceSource) |
//| - no MA10, no second series, no confluence involved at all.       |
//|                                                                    |
//| IMPORTANT: this deliberately does NOT read the pivot buffers at   |
//| shift 0. SavgolPrice's line is a centered filter - the indicator  |
//| fills the last SG_HALF (7) bars with a provisional edge estimate  |
//| that keeps getting REVISED as new bars arrive, only becoming      |
//| final once SG_HALF bars later - so a peak/valley flagged in that  |
//| window can be revised away once the underlying value changes.     |
//| Reading at shift SG_HALF is the earliest point guaranteed to      |
//| never change again. The chart's plotted arrows still show pivots  |
//| in that live tail for responsiveness - only the trading decision  |
//| waits for it to settle.                                           |
//+------------------------------------------------------------------+
void CheckSavgolPivotSignal()
{
   double peakVal   = GetIndicatorValue(g_priceHandle, PRICE_BUF_PEAK,   SG_HALF);
   double valleyVal = GetIndicatorValue(g_priceHandle, PRICE_BUF_VALLEY, SG_HALF);

   if(peakVal == EMPTY_VALUE && valleyVal == EMPTY_VALUE)
      return; // no pivot at the earliest bar guaranteed to be final

   datetime signalTime = iTime(_Symbol, PERIOD_CURRENT, SG_HALF);
   Print("Savgol EA PV: t=", TimeToString(signalTime),
         " peak=", peakVal, " valley=", valleyVal,
         " lastExec=", TimeToString(g_lastExecutedEntryTime));

   if(signalTime == 0 || signalTime <= g_lastExecutedEntryTime)
      return; // already handled, or predates EA attach

   int side = (peakVal != EMPTY_VALUE) ? 1 : 0; // peak -> SHORT, valley -> LONG

   if(ProcessNoStopsEntry(side, signalTime))
      g_lastExecutedEntryTime = signalTime;
   // else: leave g_lastExecutedEntryTime alone so this signal is retried next bar
}

//+------------------------------------------------------------------+
void CheckSignals()
{
   int available = Bars(_Symbol, PERIOD_CURRENT);
   int n = MathMin(InpLookbackBars, available);
   if(n < InpMAPeriod + 20)
      return;

   double bufMA[], bufMAPeak[], bufMAValley[], bufSCPeak[], bufSCValley[];
   ArraySetAsSeries(bufMA,       true);
   ArraySetAsSeries(bufMAPeak,   true);
   ArraySetAsSeries(bufMAValley, true);
   ArraySetAsSeries(bufSCPeak,   true);
   ArraySetAsSeries(bufSCValley, true);

   if(CopyBuffer(g_handle, BUF_MA10,            0, n, bufMA)       <= 0) return;
   if(CopyBuffer(g_handle, BUF_MA10_PEAK,       0, n, bufMAPeak)   <= 0) return;
   if(CopyBuffer(g_handle, BUF_MA10_VALLEY,     0, n, bufMAValley) <= 0) return;
   if(CopyBuffer(g_handle, BUF_SAVGOL_C_PEAK,   0, n, bufSCPeak)   <= 0) return;
   if(CopyBuffer(g_handle, BUF_SAVGOL_C_VALLEY, 0, n, bufSCValley) <= 0) return;

   datetime timeSeries[];
   ArraySetAsSeries(timeSeries, true);
   if(CopyTime(_Symbol, PERIOD_CURRENT, 0, n, timeSeries) <= 0) return;

   // Convert everything to chronological (oldest-first), matching the ordering
   // TradeSavgol.py's DataFrame rows are in.
   double maPeakChrono[], maValleyChrono[], scPeakChrono[], scValleyChrono[];
   datetime timeChrono[];
   ReverseArray(bufMAPeak,   maPeakChrono,   n);
   ReverseArray(bufMAValley, maValleyChrono, n);
   ReverseArray(bufSCPeak,   scPeakChrono,   n);
   ReverseArray(bufSCValley, scValleyChrono, n);
   ReverseTimeArray(timeSeries, timeChrono, n);

   int maPeakPos[], maValleyPos[], scPeakPos[], scValleyPos[];
   CollectPivotPositions(maPeakChrono,   n, maPeakPos);
   CollectPivotPositions(maValleyChrono, n, maValleyPos);
   CollectPivotPositions(scPeakChrono,   n, scPeakPos);
   CollectPivotPositions(scValleyChrono, n, scValleyPos);

   int entryPos[], entrySide[];
   // LONG: MA10 valley + Savgol(close) valley
   MatchSide(maValleyPos, scValleyPos, InpMaxGapBars, InpEntryDelayBars, n, entryPos, 0, entrySide);
   // SHORT: MA10 peak + Savgol(close) peak
   MatchSide(maPeakPos,   scPeakPos,   InpMaxGapBars, InpEntryDelayBars, n, entryPos, 1, entrySide);

   int count = ArraySize(entryPos);
   if(count == 0)
      return;

   // Sort entries by entryPos ascending (simple insertion sort - count is small).
   for(int i = 1; i < count; i++)
   {
      int posVal = entryPos[i];
      int sideVal = entrySide[i];
      int j = i - 1;
      while(j >= 0 && entryPos[j] > posVal)
      {
         entryPos[j + 1]  = entryPos[j];
         entrySide[j + 1] = entrySide[j];
         j--;
      }
      entryPos[j + 1]  = posVal;
      entrySide[j + 1] = sideVal;
   }

   for(int i = 0; i < count; i++)
   {
      datetime entryTime = timeChrono[entryPos[i]];
      if(entryTime > g_lastExecutedEntryTime)
      {
         if(!ProcessNewEntry(entrySide[i], entryTime))
            break; // stop here and retry this same signal next bar, rather than
                    // silently skipping to later ones and losing it
         g_lastExecutedEntryTime = entryTime;
      }
   }
}

//+------------------------------------------------------------------+
void OnTick()
{
   if(!IsNewBar())
      return;

   // If a previous bar's close queued a reverse entry, this bar's job is to
   // open it (and only that) - never evaluate fresh signals on the same bar
   // a reverse entry is being opened.
   if(TryFulfillPendingEntry())
      return;

   if(InpSignalMode == SIGNAL_CONFLUENCE)
      CheckSignals();
   else
      CheckSavgolPivotSignal();
}
