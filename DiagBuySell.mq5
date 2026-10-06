//+------------------------------------------------------------------+
//| DiagBuySell.mq5 - throwaway diagnostic EA to dump how many        |
//| non-empty Buy/Sell values Savgol_Peak_Valley actually computes.  |
//+------------------------------------------------------------------+
#property version "1.00"

int g_handle = INVALID_HANDLE;
bool g_done = false;

int OnInit()
{
   g_handle = iCustom(_Symbol, PERIOD_CURRENT, "Savgol_Peak_Valley", 10, 3, 2, true, false, false);
   if(g_handle == INVALID_HANDLE)
   {
      Print("DIAG: iCustom FAILED, error=", GetLastError());
      return(INIT_FAILED);
   }
   Print("DIAG: iCustom handle OK");
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   if(g_done) return;

   int bars = Bars(_Symbol, PERIOD_CURRENT);
   if(bars < 50) return;

   int n = MathMin(bars, 2000);

   double buy[], sell[], maPeak[], maValley[], scPeak[], scValley[];
   ArraySetAsSeries(buy, true);
   ArraySetAsSeries(sell, true);
   ArraySetAsSeries(maPeak, true);
   ArraySetAsSeries(maValley, true);
   ArraySetAsSeries(scPeak, true);
   ArraySetAsSeries(scValley, true);

   int cBuy   = CopyBuffer(g_handle, 7, 0, n, buy);
   int cSell  = CopyBuffer(g_handle, 8, 0, n, sell);
   int cMaPk  = CopyBuffer(g_handle, 3, 0, n, maPeak);
   int cMaVl  = CopyBuffer(g_handle, 4, 0, n, maValley);
   int cScPk  = CopyBuffer(g_handle, 5, 0, n, scPeak);
   int cScVl  = CopyBuffer(g_handle, 6, 0, n, scValley);

   Print("DIAG: copied counts buy=", cBuy, " sell=", cSell,
         " maPeak=", cMaPk, " maValley=", cMaVl, " scPeak=", cScPk, " scValley=", cScVl);

   int nBuy = 0, nSell = 0, nMaPeak = 0, nMaValley = 0, nScPeak = 0, nScValley = 0;
   for(int i = 0; i < n; i++)
   {
      if(cBuy  > i && buy[i]  != EMPTY_VALUE) nBuy++;
      if(cSell > i && sell[i] != EMPTY_VALUE) nSell++;
      if(cMaPk > i && maPeak[i]   != EMPTY_VALUE) nMaPeak++;
      if(cMaVl > i && maValley[i] != EMPTY_VALUE) nMaValley++;
      if(cScPk > i && scPeak[i]   != EMPTY_VALUE) nScPeak++;
      if(cScVl > i && scValley[i] != EMPTY_VALUE) nScValley++;
   }

   Print("DIAG counts over last ", n, " bars: Buy=", nBuy, " Sell=", nSell,
         "  MA10Peak=", nMaPeak, " MA10Valley=", nMaValley,
         "  SavgolClosePeak=", nScPeak, " SavgolCloseValley=", nScValley);

   // Print a few sample non-empty Buy entries (if any) with their bar time.
   int shown = 0;
   for(int i = 0; i < n && shown < 5; i++)
   {
      if(cBuy > i && buy[i] != EMPTY_VALUE)
      {
         datetime t = iTime(_Symbol, PERIOD_CURRENT, i);
         Print("DIAG Buy sample: bar=", i, " time=", TimeToString(t), " value=", buy[i]);
         shown++;
      }
   }
   if(shown == 0)
      Print("DIAG: no non-empty Buy values found in the copied window.");

   g_done = true;
}
