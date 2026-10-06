//+------------------------------------------------------------------+
//|                                                       Savgol.mq5 |
//|                                                                  |
//| Ports the series TradeSavgol.py's entry signal actually depends  |
//| on. find_entry_signals() in TradeSavgol.py fires a LONG/SHORT    |
//| signal when a raw MA10 pivot and a Savitzky-Golay(close) pivot   |
//| (SAVGOL15_close) land within max_gap=3 bars of each other:       |
//|   MA10 valley + Savgol(close) valley -> LONG                     |
//|   MA10 peak   + Savgol(close) peak   -> SHORT                    |
//| This indicator exposes both series and both sets of raw pivots   |
//| so an EA can reproduce that exact confluence rule via iCustom().  |
//|                                                                  |
//| Savgol(MA10) is also plotted (matching the 3rd panel of the      |
//| Python report / the Savgol_v1 indicator) but is NOT used by      |
//| TradeSavgol.py's signal logic - it carries no pivot arrows here.  |
//|                                                                  |
//| The 15-tap centered filter weights below were computed once in  |
//| Python via scipy.signal.savgol_coeffs(15, 3) and are fixed for   |
//| window=15, polyorder=3 (SAVGOL_WINDOW/SAVGOL_POLYORDER in        |
//| TradeSavgol.py). Changing InpMAPeriod does NOT require new       |
//| coefficients; changing the SG window/polyorder does - MQL5 has  |
//| no built-in least-squares solver, so those would need to be      |
//| recomputed in Python and pasted in again.                        |
//|                                                                  |
//| Pivot arrows mark raw peaks/valleys only - the 2-bar-after-later-|
//| pivot entry delay and the max_gap confluence matching from       |
//| TradeSavgol.py are NOT applied here; the EA built on top of this |
//| indicator's buffers reproduces that itself.                      |
//+------------------------------------------------------------------+
#property copyright "Savgol"
#property version   "2.00"
#property indicator_chart_window
#property indicator_buffers 7
#property indicator_plots   7

#property indicator_label1  "MA10"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrSaddleBrown
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2

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

input int    InpMAPeriod   = 10;   // Moving average period (MA_WINDOW in TradeSavgol.py)
input double InpMinPipMove = 1.0;  // minimum move (in pips) for a peak/valley to count as real, not float noise

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
// REVERSED, applied to the LAST 15 samples (this symmetry was verified
// empirically against scipy's own impulse response - trailing output
// N-1-k = dot(reverse(SG_EDGE_COEF[k]), src[N-15..N-1])).
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
//| default, no prominence/distance filter) - except both sides of    |
//| the comparison must differ by more than one pip (InpMinPipMove),   |
//| so a peak/valley isn't registered from sub-pip floating-point      |
//| noise. Bar i is only knowable once bar i+1 exists, so the last    |
//| confirmable index is n-2 - but within the valid range              |
//| [firstValid+1, lastValid-1] of src.                                |
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

int OnInit()
{
   SetIndexBuffer(0, BufMA,                INDICATOR_DATA);
   SetIndexBuffer(1, BufSavgolClose,        INDICATOR_DATA);
   SetIndexBuffer(2, BufSavgolMA,           INDICATOR_DATA);
   SetIndexBuffer(3, BufMAPeak,             INDICATOR_DATA);
   SetIndexBuffer(4, BufMAValley,           INDICATOR_DATA);
   SetIndexBuffer(5, BufSavgolClosePeak,    INDICATOR_DATA);
   SetIndexBuffer(6, BufSavgolCloseValley,  INDICATOR_DATA);

   PlotIndexSetInteger(3, PLOT_ARROW, 234); // Wingdings down arrow
   PlotIndexSetInteger(4, PLOT_ARROW, 233); // Wingdings up arrow
   PlotIndexSetInteger(5, PLOT_ARROW, 234);
   PlotIndexSetInteger(6, PLOT_ARROW, 233);

   for(int p = 0; p < 7; p++)
      PlotIndexSetDouble(p, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   IndicatorSetString(INDICATOR_SHORTNAME, "Savgol(" + IntegerToString(InpMAPeriod) + ")");
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

   // --- Pivots that feed TradeSavgol.py's entry confluence ---
   // Neither series has a right-edge trim any more (edge coefficients cover
   // it), so both simply run up to the last available bar.
   int maLastValid = rates_total - 1;
   FindPivots(BufMA, BufMAPeak, BufMAValley, rates_total, start, InpMAPeriod - 1, maLastValid);

   int closeLastValid = rates_total - 1;
   FindPivots(BufSavgolClose, BufSavgolClosePeak, BufSavgolCloseValley,
              rates_total, start, closeFirstValid, closeLastValid);

   return(rates_total);
}
