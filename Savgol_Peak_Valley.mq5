//+------------------------------------------------------------------+
//|                                       Savgol_Peak_Valley.mq5     |
//|                                                                  |
//| Plots Buy/Sell confluence signals on price (default), with       |
//| optional overlays of the raw find_peaks/valley pivots and the    |
//| Savitzky-Golay line(s) they are matched against. The peak/valley |
//| detection input series is MA10 (InpMAPeriod, changeable).        |
//|                                                                  |
//| The concept of find_peaks and valleys is based on the idea that local maxima (peaks) and minima (valleys)
//| in a time series can indicate potential turning points or areas of interest for analysis. By identifying
//| these points, one can gain insights into the behavior of the data and make informed decisions based on the observed patterns.
//|
//| Whenever a peak is detected by both find_peaks and the Savitzky-Golay filter to be within
//| a certain proximity (3 frames, should be input), it suggests that the peak is a significant feature of
//| the data, as it is confirmed by two different methods. This can increase confidence in the validity of the detected peak and
//| its potential relevance for further analysis or decision-making. It suggests that the peak is a significant feature of the data, as it is
//| confirmed by two different methods. This can increase confidence in the validity of the detected peak and its potential
//| relevance for further analysis or decision-making.
//|
//| Opposite logic applies for valleys, where a valley detected by both methods within the specified proximity
//| indicates a significant local minimum in the data.
//|
//| Long/short signals are generated based on the confluence of peaks and valleys detected by both methods.
//| Order are executed with a 2-frame delay after the later of the two pivots, and exit conditions are
//| based on stop loss, take profit, or opposite signals.
//|
//| (This indicator itself only plots the Buy/Sell confluence signal   |
//| markers - it does not place orders or apply stop loss/take        |
//| profit/opposite-signal exits. See the Savgol EA for that.)        |
//+------------------------------------------------------------------+
#property copyright "Savgol_Peak_Valley"
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 9
#property indicator_plots   9

#property indicator_label1  "MA10"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrSilver
#property indicator_style1  STYLE_DOT
#property indicator_width1  1

#property indicator_label2  "Savgol(close)"
#property indicator_type2   DRAW_LINE
#property indicator_color2  clrDodgerBlue
#property indicator_width2  2

#property indicator_label3  "Savgol(MA10)"
#property indicator_type3   DRAW_LINE
#property indicator_color3  clrOrange
#property indicator_width3  1

#property indicator_label4  "MA10 Peak"
#property indicator_type4   DRAW_ARROW
#property indicator_color4  clrRed
#property indicator_width4  2

#property indicator_label5  "MA10 Valley"
#property indicator_type5   DRAW_ARROW
#property indicator_color5  clrLimeGreen
#property indicator_width5  2

#property indicator_label6  "Savgol(close) Peak"
#property indicator_type6   DRAW_ARROW
#property indicator_color6  clrMagenta
#property indicator_width6  2

#property indicator_label7  "Savgol(close) Valley"
#property indicator_type7   DRAW_ARROW
#property indicator_color7  clrAqua
#property indicator_width7  2

#property indicator_label8  "Buy"
#property indicator_type8   DRAW_ARROW
#property indicator_color8  clrBlue
#property indicator_width8  3

#property indicator_label9  "Sell"
#property indicator_type9   DRAW_ARROW
#property indicator_color9  clrRed
#property indicator_width9  3

input int  InpMAPeriod      = 10;    // MA period feeding both the raw and Savgol(MA10) pivot detection
input int  InpMaxGapBars    = 3;     // max bars between an MA10 pivot and a Savgol(close) pivot to count as confluence
input int  InpEntryDelayBars= 2;     // bars after the later pivot before the Buy/Sell signal fires
input bool InpShowBuySell   = true;  // plot Buy/Sell confluence signals on price (default)
input bool InpShowPivots    = false; // plot the raw find_peaks MA10/Savgol(close) peak & valley markers
input bool InpShowSavgol    = false; // plot the Savgol(close) and Savgol(MA10) lines
input double InpMinPipMove  = 1.0;   // minimum move (in pips) for a peak/valley to count as real, not float noise

//+------------------------------------------------------------------+
//| One pip in price terms for the current symbol (10 points on a     |
//| 3/5-digit broker, 1 point otherwise) - used so peak/valley         |
//| comparisons require a real move, not sub-pip floating-point noise.|
//+------------------------------------------------------------------+
double PipSize()
{
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   return (digits == 3 || digits == 5) ? point * 10.0 : point;
}

#define SG_WINDOW 15
#define SG_HALF   7
double SG_COEF[SG_WINDOW] =
{
   -0.07058823529411806, -0.011764705882352865,  0.03800904977375594,
    0.07873303167420864,  0.11040723981900519,   0.13303167420814557,
    0.14660633484162983,  0.15113122171945792,   0.14660633484162983,
    0.13303167420814557,  0.11040723981900517,   0.07873303167420861,
    0.03800904977375586, -0.011764705882353073, -0.07058823529411816
};

// scipy.signal.savgol_filter's default mode='interp' does NOT leave NaN at
// the edges - it fits an edge-anchored window and evaluates it AT the edge
// point, which is what TradeSavgol.py's add_savgol() actually produces.
// SG_EDGE_COEF[k] (k=0..6) is the weight row for evaluating the point at
// offset k within a 15-sample window anchored at the data boundary
// (scipy.signal.savgol_coeffs(15, 3, pos=k)), applied to the FIRST 15
// samples for the leading edge. The trailing edge uses these same rows
// REVERSED, applied to the LAST 15 samples (verified empirically against
// scipy's own impulse response).
double SG_EDGE_COEF[SG_HALF][SG_WINDOW] =
{
   {0.67287581699346299, 0.37385620915032441, 0.15816993464052284, 0.014379084967321848, -0.068954248366011231, -0.1032679738562072, -0.099999999999998562, -0.070588235294117146, -0.026470588235294384, 0.020915032679737523, 0.06013071895424614, 0.079738562091501305, 0.068300653594769889, 0.014379084967320844, -0.09346405228757905},
   {0.3738562091503258, 0.27231559290382762, 0.18898225957049508, 0.12222222222222322, 0.070401493930906542, 0.031886087768441596, 0.0050420168067232679, -0.011764705882352858, -0.02016806722689108, -0.021802054154995996, -0.018300653594772148, -0.011297852474323758, -0.0024276377217560124, 0.0066760037348274886, 0.014379084967321598},
   {0.15816993464052209, 0.18898225957049544, 0.19646986999928212, 0.18503196150255027, 0.15906772965596511, 0.12297637003519371, 0.081157078215901735, 0.038009049773755466, -0.0020685197155788822, -0.034676434676435032, -0.055415499533146617, -0.059886518710047923, -0.043690296631473252, -0.0024276377217553862, 0.068300653594771651},
   {0.014379084967319655, 0.12222222222222345, 0.18503196150255011, 0.21009839833369268, 0.20471162824103972, 0.17616174674998167, 0.13173884938590788, 0.07873303167420781, 0.02443438914027115, -0.023866982690512219, -0.058880988292752498, -0.073317532141060834, -0.059886518710047992, -0.011297852474323225, 0.079738562091503082},
   {-0.068954248366013271, 0.070401493930907055, 0.15906772965596522, 0.20471162824103997, 0.21500035911800569, 0.19760109171873816, 0.16018099547511264, 0.11040723981900416, 0.05594699418228799, 0.0044674279968398359, -0.036364289305465106, -0.058880988292751971, -0.05541549953314634, -0.018300653594771468, 0.060130718954247805},
   {-0.10326797385620912, 0.031886087768441984, 0.12297637003519368, 0.17616174674998181, 0.19760109171873824, 0.19345327874739568, 0.169877181641887, 0.13303167420814446, 0.089075630252100579, 0.044167923579688483, 0.0044674279968403702, -0.02386698269051106, -0.034676434676434414, -0.021802054154995559, 0.020915032679737891},
   {-0.099999999999999534, 0.0050420168067235038, 0.081157078215901721, 0.13173884938590802, 0.16018099547511255, 0.16987718164188684, 0.16422107304460212, 0.14660633484162869, 0.12042663219133784, 0.089075630252101079, 0.055946994182288698, 0.024434389140272419, -0.0020685197155783375, -0.020168067226890928, -0.026470588235294648}
};

double BufMA[];
double BufSavgolClose[];
double BufSavgolMA[];
double BufMAPeak[];
double BufMAValley[];
double BufSavgolClosePeak[];
double BufSavgolCloseValley[];
double BufBuy[];
double BufSell[];

int g_lastConfluenceRatesTotal = -1;

//+------------------------------------------------------------------+
//| Savitzky-Golay smoothing of src into dst, matching scipy's        |
//| mode='interp' exactly: interior points use the centered 15-tap    |
//| filter, and the first/last SG_HALF points (once at least one full |
//| window's worth of data exists from firstValid) use the edge       |
//| coefficients instead of being left empty. dst is EMPTY_VALUE only |
//| when there isn't yet a single full window of data.                |
//+------------------------------------------------------------------+
void ApplySavgol(const double &src[], double &dst[], int n, int start, int firstValid)
{
   if(n - firstValid < SG_WINDOW)
   {
      for(int i = MathMax(start, firstValid); i < n; i++)
         dst[i] = EMPTY_VALUE;
      return;
   }

   // Interior: centered filter.
   int interiorStart = MathMax(start, firstValid + SG_HALF);
   int interiorLast  = n - SG_HALF - 1;
   for(int i = interiorStart; i <= interiorLast; i++)
   {
      double acc = 0;
      for(int k = -SG_HALF; k <= SG_HALF; k++)
         acc += SG_COEF[k + SG_HALF] * src[i + k];
      dst[i] = acc;
   }

   // Leading edge: first SG_HALF points, fixed window [firstValid..firstValid+14].
   for(int k = 0; k < SG_HALF; k++)
   {
      int i = firstValid + k;
      if(i < start) continue; // already correct from an earlier call
      double acc = 0;
      for(int j = 0; j < SG_WINDOW; j++)
         acc += SG_EDGE_COEF[k][j] * src[firstValid + j];
      dst[i] = acc;
   }

   // Trailing edge: last SG_HALF points, fixed window [n-15..n-1] (shifts every
   // new bar, so always recomputed - it's only 7*15 multiplications).
   for(int k = 0; k < SG_HALF; k++)
   {
      int i = n - 1 - k;
      double acc = 0;
      for(int j = 0; j < SG_WINDOW; j++)
         acc += SG_EDGE_COEF[k][SG_WINDOW - 1 - j] * src[n - SG_WINDOW + j]; // reversed row
      dst[i] = acc;
   }
}

//+------------------------------------------------------------------+
//| Strict local-max/min pivots of src (matches scipy.find_peaks'     |
//| default, no prominence/distance filter). Bar i is only knowable   |
//| once bar i+1 exists.                                              |
//+------------------------------------------------------------------+
void FindPivots(const double &src[], double &peakBuf[], double &valleyBuf[],
                 int n, int start, int firstValid, int lastValid)
{
   double minMove = InpMinPipMove * PipSize();
   int pivotStart = MathMax(start, firstValid + 1);
   int pivotEnd   = lastValid - 1;
   for(int i = pivotStart; i <= pivotEnd; i++)
   {
      peakBuf[i]   = EMPTY_VALUE;
      valleyBuf[i] = EMPTY_VALUE;
      double prevVal = src[i - 1];
      double currVal = src[i];
      double nextVal = src[i + 1];
      if(currVal - prevVal > minMove && currVal - nextVal > minMove)
         peakBuf[i] = currVal;
      else if(prevVal - currVal > minMove && nextVal - currVal > minMove)
         valleyBuf[i] = currVal;
   }
   for(int i = MathMax(pivotEnd + 1, 0); i < n; i++)
   {
      peakBuf[i]   = EMPTY_VALUE;
      valleyBuf[i] = EMPTY_VALUE;
   }
}

//+------------------------------------------------------------------+
//| Chronological indices where buf[i] != EMPTY_VALUE                 |
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
//| Greedy confluence matching for one side - direct port of          |
//| TradeSavgol.py's find_entry_signals(): for each MA10 pivot         |
//| (chronological), pick the closest not-yet-used Savgol(close)      |
//| pivot within InpMaxGapBars; the signal fires InpEntryDelayBars     |
//| after the later of the two.                                       |
//+------------------------------------------------------------------+
void MatchSide(const int &maPositions[], const int &sigPositions[], int n, bool isLong,
               const double &low[], const double &high[], double &buyBuf[], double &sellBuf[])
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
         if(dist <= InpMaxGapBars && dist < bestDist)
         {
            bestDist = dist;
            bestIdx  = b;
         }
      }
      if(bestIdx < 0) continue;
      used[bestIdx] = true;

      int pivotPos = MathMax(maPos, sigPositions[bestIdx]);
      int entryPos = pivotPos + InpEntryDelayBars;
      if(entryPos >= n) continue;

      // Push the arrow visibly clear of the candle (half that bar's range) so a
      // Buy arrow at the low doesn't blend into a bullish candle's wick/body.
      double clearance = (high[entryPos] - low[entryPos]) * 0.5;
      if(isLong)
         buyBuf[entryPos] = low[entryPos] - clearance;
      else
         sellBuf[entryPos] = high[entryPos] + clearance;
   }
}

//+------------------------------------------------------------------+
void RebuildConfluenceSignals(int n, const double &low[], const double &high[])
{
   ArrayInitialize(BufBuy,  EMPTY_VALUE);
   ArrayInitialize(BufSell, EMPTY_VALUE);

   int maPeakPos[], maValleyPos[], scPeakPos[], scValleyPos[];
   CollectPivotPositions(BufMAPeak,            n, maPeakPos);
   CollectPivotPositions(BufMAValley,          n, maValleyPos);
   CollectPivotPositions(BufSavgolClosePeak,   n, scPeakPos);
   CollectPivotPositions(BufSavgolCloseValley, n, scValleyPos);

   MatchSide(maValleyPos, scValleyPos, n, true,  low, high, BufBuy, BufSell); // LONG: valley + valley
   MatchSide(maPeakPos,   scPeakPos,   n, false, low, high, BufBuy, BufSell); // SHORT: peak + peak
}

//+------------------------------------------------------------------+
int OnInit()
{
   SetIndexBuffer(0, BufMA,                INDICATOR_DATA);
   SetIndexBuffer(1, BufSavgolClose,        INDICATOR_DATA);
   SetIndexBuffer(2, BufSavgolMA,           INDICATOR_DATA);
   SetIndexBuffer(3, BufMAPeak,             INDICATOR_DATA);
   SetIndexBuffer(4, BufMAValley,           INDICATOR_DATA);
   SetIndexBuffer(5, BufSavgolClosePeak,    INDICATOR_DATA);
   SetIndexBuffer(6, BufSavgolCloseValley,  INDICATOR_DATA);
   SetIndexBuffer(7, BufBuy,                INDICATOR_DATA);
   SetIndexBuffer(8, BufSell,               INDICATOR_DATA);

   PlotIndexSetInteger(3, PLOT_ARROW, 234); // Wingdings down arrow
   PlotIndexSetInteger(4, PLOT_ARROW, 233); // Wingdings up arrow
   PlotIndexSetInteger(5, PLOT_ARROW, 234);
   PlotIndexSetInteger(6, PLOT_ARROW, 233);
   PlotIndexSetInteger(7, PLOT_ARROW, 233); // Buy: up arrow
   PlotIndexSetInteger(8, PLOT_ARROW, 234); // Sell: down arrow

   for(int p = 0; p < 9; p++)
      PlotIndexSetDouble(p, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   // Optional overlays: hide the plot entirely when its input flag is off.
   if(!InpShowSavgol)
   {
      PlotIndexSetInteger(1, PLOT_DRAW_TYPE, DRAW_NONE);
      PlotIndexSetInteger(2, PLOT_DRAW_TYPE, DRAW_NONE);
   }
   if(!InpShowPivots)
   {
      PlotIndexSetInteger(3, PLOT_DRAW_TYPE, DRAW_NONE);
      PlotIndexSetInteger(4, PLOT_DRAW_TYPE, DRAW_NONE);
      PlotIndexSetInteger(5, PLOT_DRAW_TYPE, DRAW_NONE);
      PlotIndexSetInteger(6, PLOT_DRAW_TYPE, DRAW_NONE);
   }
   if(!InpShowBuySell)
   {
      PlotIndexSetInteger(7, PLOT_DRAW_TYPE, DRAW_NONE);
      PlotIndexSetInteger(8, PLOT_DRAW_TYPE, DRAW_NONE);
   }

   IndicatorSetString(INDICATOR_SHORTNAME, "Savgol_Peak_Valley(" + IntegerToString(InpMAPeriod) + ")");
   IndicatorSetInteger(INDICATOR_DIGITS, _Digits);

   return(INIT_SUCCEEDED);
}

int OnCalculate(const int rates_total,
                 const int prev_calculated,
                 const datetime &time[],
                 const double &open[],
                 const double &high[],
                 const double &low[],
                 const double &close[],
                 const long &tick_volume[],
                 const long &volume[],
                 const int &spread[])
{
   if(rates_total < InpMAPeriod + SG_WINDOW)
      return 0;

   if(prev_calculated == 0)
   {
      ArrayInitialize(BufMA,               EMPTY_VALUE);
      ArrayInitialize(BufSavgolClose,       EMPTY_VALUE);
      ArrayInitialize(BufSavgolMA,          EMPTY_VALUE);
      ArrayInitialize(BufMAPeak,            EMPTY_VALUE);
      ArrayInitialize(BufMAValley,          EMPTY_VALUE);
      ArrayInitialize(BufSavgolClosePeak,   EMPTY_VALUE);
      ArrayInitialize(BufSavgolCloseValley, EMPTY_VALUE);
      ArrayInitialize(BufBuy,               EMPTY_VALUE);
      ArrayInitialize(BufSell,              EMPTY_VALUE);
   }

   // Recompute a generous trailing window each tick: MA -> Savgol -> pivot is a
   // multi-stage chain, so a minimal 1-bar incremental update can leave stale
   // pivot markers near the right edge as new bars arrive.
   int start = (prev_calculated > 1) ? prev_calculated - (SG_WINDOW + 5) : 0;
   if(start < InpMAPeriod - 1)
      start = InpMAPeriod - 1;

   // --- MA10: simple moving average of close ---
   for(int i = start; i < rates_total; i++)
   {
      double sum = 0;
      for(int k = 0; k < InpMAPeriod; k++)
         sum += close[i - k];
      BufMA[i] = sum / InpMAPeriod;
   }

   // --- Savgol(close): 15-tap filter on close, edge-corrected like scipy's mode='interp' ---
   int closeFirstValid = 0;
   ApplySavgol(close, BufSavgolClose, rates_total, start, closeFirstValid);

   // --- Savgol(MA10): same filter on BufMA (reference line only) ---
   int maFirstValid = InpMAPeriod - 1;
   ApplySavgol(BufMA, BufSavgolMA, rates_total, start, maFirstValid);

   // --- Pivots that feed the Buy/Sell confluence ---
   // Neither series has a right-edge trim any more (edge coefficients cover
   // it), so both simply run up to the last available bar.
   int maLastValid = rates_total - 1;
   FindPivots(BufMA, BufMAPeak, BufMAValley, rates_total, start, InpMAPeriod - 1, maLastValid);

   int closeLastValid = rates_total - 1;
   FindPivots(BufSavgolClose, BufSavgolClosePeak, BufSavgolCloseValley,
              rates_total, start, closeFirstValid, closeLastValid);

   // Confluence pivots can only change when new bars close (a forming bar can
   // never itself be a confirmed pivot), so only re-run the matching when the
   // bar count actually grows - not on every intra-bar tick.
   if(rates_total != g_lastConfluenceRatesTotal)
   {
      RebuildConfluenceSignals(rates_total, low, high);
      g_lastConfluenceRatesTotal = rates_total;
   }

   return(rates_total);
}
