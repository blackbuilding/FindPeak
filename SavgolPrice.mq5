//+------------------------------------------------------------------+
//|                                                SavgolPrice.mq5   |
//|                                                                  |
//| Slimmed-down Savgol indicator: computes ONE Savitzky-Golay        |
//| smoothed price series and its peaks/valleys only - no MA10, no    |
//| second series, no confluence matching (see "Savgol" indicator/    |
//| Savgol.mq5 for that). The price fed into the filter is selectable |
//| (InpPriceSource): Close, Open, High, Low, or OHLC/4 (average of   |
//| Open+High+Low+Close).                                             |
//|                                                                  |
//| Same 15-tap Savitzky-Golay filter (window=15, polyorder=3) and    |
//| pivot rule (strict local max/min, min move InpMinPipMove) as the  |
//| "Savgol" indicator's Savgol(close) series - see Savgol.mq5 for    |
//| the coefficient derivation notes. Edge bars (last SG_HALF) are    |
//| provisional estimates that keep revising as new bars arrive, only |
//| becoming final once SG_HALF bars later - same caveat as Savgol.   |
//| mq5, a consumer (e.g. SavgolEA.mq5) should read shift SG_HALF or   |
//| later before treating a peak/valley as settled.                   |
//+------------------------------------------------------------------+
#property copyright "Savgol"
#property version   "1.00"
#property indicator_chart_window
#property indicator_buffers 3
#property indicator_plots   3

#property indicator_label1  "Savgol(price)"
#property indicator_type1   DRAW_LINE
#property indicator_color1  clrDodgerBlue
#property indicator_width1  2

#property indicator_label2  "Peak"
#property indicator_type2   DRAW_ARROW
#property indicator_color2  clrRed
#property indicator_width2  2

#property indicator_label3  "Valley"
#property indicator_type3   DRAW_ARROW
#property indicator_color3  clrLimeGreen
#property indicator_width3  2

enum ENUM_SAVGOL_PRICE_SOURCE
{
   SAVGOL_PRICE_CLOSE = 0, // Close
   SAVGOL_PRICE_OPEN  = 1, // Open
   SAVGOL_PRICE_HIGH  = 2, // High
   SAVGOL_PRICE_LOW   = 3, // Low
   SAVGOL_PRICE_OHLC4 = 4  // (Open+High+Low+Close)/4
};

input ENUM_SAVGOL_PRICE_SOURCE InpPriceSource = SAVGOL_PRICE_CLOSE; // price fed into the Savgol filter
input double InpMinPipMove = 1.0; // minimum move (in pips) for a peak/valley to count as real, not float noise

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
// point. SG_EDGE_COEF[k] (k=0..6) is the weight row for evaluating the point
// at offset k within a 15-sample window anchored at the data boundary
// (scipy.signal.savgol_coeffs(15, 3, pos=k)), applied to the FIRST 15
// samples for the leading edge. The trailing edge uses these same rows
// REVERSED, applied to the LAST 15 samples (verified empirically against
// scipy's own impulse response - see Savgol.mq5).
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

double BufLine[];
double BufPeak[];
double BufValley[];

//+------------------------------------------------------------------+
//| Savitzky-Golay smoothing of src into dst, matching scipy's        |
//| mode='interp' exactly: interior points use the centered 15-tap    |
//| filter, and the first/last SG_HALF points use edge coefficients   |
//| instead of being left empty. Identical to Savgol.mq5's version.   |
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
//| [firstValid+1, lastValid-1] of src. Identical to Savgol.mq5.       |
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
   SetIndexBuffer(0, BufLine,   INDICATOR_DATA);
   SetIndexBuffer(1, BufPeak,   INDICATOR_DATA);
   SetIndexBuffer(2, BufValley, INDICATOR_DATA);

   PlotIndexSetInteger(1, PLOT_ARROW, 234); // Wingdings down arrow
   PlotIndexSetInteger(2, PLOT_ARROW, 233); // Wingdings up arrow

   for(int p = 0; p < 3; p++)
      PlotIndexSetDouble(p, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   string srcName;
   switch(InpPriceSource)
   {
      case SAVGOL_PRICE_OPEN:  srcName = "open";  break;
      case SAVGOL_PRICE_HIGH:  srcName = "high";  break;
      case SAVGOL_PRICE_LOW:   srcName = "low";   break;
      case SAVGOL_PRICE_OHLC4: srcName = "ohlc4"; break;
      default:                 srcName = "close"; break;
   }
   IndicatorSetString(INDICATOR_SHORTNAME, "SavgolPrice(" + srcName + ")");
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
   if(rates_total < SG_WINDOW)
      return 0;

   if(prev_calculated == 0)
   {
      ArrayInitialize(BufLine,   EMPTY_VALUE);
      ArrayInitialize(BufPeak,   EMPTY_VALUE);
      ArrayInitialize(BufValley, EMPTY_VALUE);
   }

   // Recompute a generous trailing window each tick: filter -> pivot is a
   // 2-stage chain, so a minimal 1-bar incremental update can leave stale
   // pivot markers near the right edge as new bars arrive.
   int start = (prev_calculated > 1) ? prev_calculated - (SG_WINDOW + 5) : 0;
   if(start < 0)
      start = 0;

   // src[] is a plain local array (not an indicator buffer), so it does NOT
   // persist between calls - fill it in full every call. That's cheap (just
   // a copy or a per-bar average), unlike the Savgol filter itself, which
   // still only recomputes over the incremental [start..end] range below via
   // BufLine/BufPeak/BufValley (those ARE persistent indicator buffers).
   double src[];
   ArrayResize(src, rates_total);
   switch(InpPriceSource)
   {
      case SAVGOL_PRICE_OPEN:
         for(int i = 0; i < rates_total; i++) src[i] = open[i];
         break;
      case SAVGOL_PRICE_HIGH:
         for(int i = 0; i < rates_total; i++) src[i] = high[i];
         break;
      case SAVGOL_PRICE_LOW:
         for(int i = 0; i < rates_total; i++) src[i] = low[i];
         break;
      case SAVGOL_PRICE_OHLC4:
         for(int i = 0; i < rates_total; i++) src[i] = (open[i] + high[i] + low[i] + close[i]) / 4.0;
         break;
      default: // SAVGOL_PRICE_CLOSE
         for(int i = 0; i < rates_total; i++) src[i] = close[i];
         break;
   }

   ApplySavgol(src, BufLine, rates_total, start, 0);
   FindPivots(BufLine, BufPeak, BufValley, rates_total, start, 0, rates_total - 1);

   return(rates_total);
}
