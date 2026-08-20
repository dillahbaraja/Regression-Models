//+------------------------------------------------------------------+
//|                                            Regression_EA.mq5     |
//|   Ensemble regression trader using full-data runtime artifacts   |
//+------------------------------------------------------------------+
#property strict
#property copyright "Copyright 2026, Abdillah Baradja"
#property link      "dillahbaraja@gmail.com"
#property version   "1.00"

#include <Trade\Trade.mqh>

input ulong           InpMagicNumber           = 123456;
input int             InpMaxSpread             = 20;
input int             InpStopLoss              = 200;
input int             InpTakeProfit            = 400;
input double          InpRiskPercent           = 5.0;
input int             InpMaxOpenPosition       = 1;
input bool            InpAllowReverse          = true;
input bool            InpTradeOnlyNewBar       = true;
input bool            InpUseModelThresholds    = true;
input double          InpThresholdBuy          = 0.00015;
input double          InpThresholdSell         = -0.00015;
input bool            InpEnableDebugLog        = true;
input bool            InpEnableDebugCsv        = true;
input string          InpDebugCsvFileName      = "Regression_EA_debug.csv";
input bool            InpTraceEventWindows     = true;
input int             InpDebugStatusEveryBars  = 250;
input int             InpNewsReloadSeconds     = 300;

CTrade trade;

long onnx_handle = INVALID_HANDLE;
long ensemble_handles[4];
string ensemble_model_names[4] = {"xgboost", "lightgbm", "catboost", "sklearn_hgb"};
string gFeatureNames[];
double gScalerMeans[];
double gScalerScales[];

long gRsiHandle = INVALID_HANDLE;
long gAtrHandle = INVALID_HANDLE;
long gMacdHandle = INVALID_HANDLE;
long gBandsHandle = INVALID_HANDLE;
long gStochHandle = INVALID_HANDLE;
long gAdxHandle = INVALID_HANDLE;
long gRsiH4Handle = INVALID_HANDLE;
long gMaH4Handle = INVALID_HANDLE;
long gRsiM15Handle = INVALID_HANDLE;
long gAtrM15Handle = INVALID_HANDLE;

datetime gLastBarTime = 0;
string gThresholdFileName = "";
string gScalerFileName = "";
string gAssetFolder = "Regression_Assets_Full\\";
string gNewsEventsFileName = "NewsEvents.csv";
int gFeatureCount = 0;
int gDebugBarCounter = 0;
bool gLoggedRatesOrder = false;
double gActiveThresholdBuy = 0.0;
double gActiveThresholdSell = 0.0;
datetime gLastNewsReloadTime = 0;
bool gNewsLoaded = false;
string gDebugCsvFileName = "";

datetime gNewsEventTimes[];
string gNewsEventNames[];
string gNewsCountryCodes[];
int gNewsImportances[];
double gNewsActuals[];
double gNewsForecasts[];
double gNewsPreviouss[];

bool IsNewBar();
bool IsEnsembleModelType();
bool IsVectorOutputModelName(const string model_name);
string ResolveModelString();
string ResolveTimeframeString();
string ResolveThresholdFileName();
string ResolveScalerFileName();
int OpenCommonCsv(const string file_name, const int flags);
int ResolveMinimumImportance();
bool InitSingleOnnx(const string file_name, const string model_name, long &handle);
bool InitOnnx();
bool InitIndicators();
bool LoadScalerParams();
bool LoadModelThresholds();
void ReleaseModels();
bool LoadNewsEvents();
bool RefreshNewsEventsIfDue(const bool force_reload=false);
bool IsRelevantCountryCode(const string country_code);
string ResolveCountryCode(const string currency_code);
bool BuildScaledFeatureVector(const datetime bar_time, vectorf &features, int &max_importance, string &event_name);
double ComputeVolatility20(MqlRates &rates[], const int shift);
bool ReadIndicatorValue(const long handle, const int buffer_index, const int shift, double &value);
bool RunPredictionForHandle(const long handle, const string model_name, const vectorf &features, double &prediction);
bool EvaluateEvent(const datetime bar_time, double &prediction, int &importance, string &event_name);
bool ProcessRuntimeWindow(const datetime previous_bar_time, const datetime current_bar_time, const int current_spread);
int CountOpenPositions(int &buy_count, int &sell_count);
void ClosePositions(ENUM_POSITION_TYPE type);
bool ExecuteSignal(const int signal, const double prediction, const string order_comment);
double CalculateOrderLotSize();
string SanitizeCsvText(const string value);
bool AppendDebugCsv(const datetime previous_bar_time,
                    const datetime current_bar_time,
                    const string event_name,
                    const int importance,
                    const double prediction,
                    const int signal,
                    const int current_spread,
                    const bool executed,
                    const string status);
void DebugPrint(const string message);
void DebugRuntimeStatus(const datetime previous_bar_time, const datetime current_bar_time, const int current_spread);

//+------------------------------------------------------------------+
bool IsNewBar()
  {
   datetime current_time = iTime(_Symbol, _Period, 0);
   if(current_time != gLastBarTime)
     {
      gLastBarTime = current_time;
      return true;
     }
  return false;
  }

//+------------------------------------------------------------------+
void DebugPrint(const string message)
  {
   if(InpEnableDebugLog)
      Print(message);
  }

//+------------------------------------------------------------------+
void DebugRuntimeStatus(const datetime previous_bar_time, const datetime current_bar_time, const int current_spread)
  {
   if(!InpEnableDebugLog || InpDebugStatusEveryBars <= 0)
      return;

   gDebugBarCounter++;
   if((gDebugBarCounter % InpDebugStatusEveryBars) != 0)
      return;

   Print(StringFormat(
      "Runtime status | previous=%s | current=%s | spread=%d | news_rows=%d | last_reload=%s",
      TimeToString(previous_bar_time, TIME_DATE | TIME_MINUTES),
      TimeToString(current_bar_time, TIME_DATE | TIME_MINUTES),
      current_spread,
      ArraySize(gNewsEventTimes),
      (gLastNewsReloadTime > 0 ? TimeToString(gLastNewsReloadTime, TIME_DATE | TIME_MINUTES) : "never")
   ));
  }

//+------------------------------------------------------------------+
bool IsEnsembleModelType()
  {
   return true;
  }

//+------------------------------------------------------------------+
bool IsVectorOutputModelName(const string model_name)
  {
   return false;
  }

//+------------------------------------------------------------------+
string ResolveModelString()
  {
   return "ensemble_regression";
  }

//+------------------------------------------------------------------+
string ResolveTimeframeString()
  {
   string timeframe_str = EnumToString(Period());
   StringReplace(timeframe_str, "PERIOD_", "");
   return timeframe_str;
  }

//+------------------------------------------------------------------+
string ResolveThresholdFileName()
  {
   return gAssetFolder + _Symbol + "_" + ResolveTimeframeString() + "_" + ResolveModelString() + "_thresholds.csv";
  }

//+------------------------------------------------------------------+
string ResolveScalerFileName()
  {
   return gAssetFolder + _Symbol + "_" + ResolveTimeframeString() + "_scaler_params.csv";
  }

//+------------------------------------------------------------------+
int OpenCommonCsv(const string file_name, const int flags)
  {
   return FileOpen(file_name, flags | FILE_COMMON, ',');
  }

//+------------------------------------------------------------------+
int ResolveMinimumImportance()
  {
   return 1;
  }

//+------------------------------------------------------------------+
bool InitSingleOnnx(const string file_name, const string model_name, long &handle)
  {
   uint flags = ONNX_USE_CPU_ONLY;
   handle = OnnxCreate(file_name, flags | ONNX_COMMON_FOLDER);

   if(handle == INVALID_HANDLE)
     {
      Print("Failed to load ONNX model: ", file_name, " Err: ", GetLastError());
      return false;
     }

   long input_shape[2];
   input_shape[0] = 1;
   input_shape[1] = gFeatureCount;

   if(!OnnxSetInputShape(handle, 0, input_shape))
     {
      Print("Failed to set ONNX input shape for ", file_name, ". Err: ", GetLastError());
      return false;
     }

   if(IsVectorOutputModelName(model_name))
     {
      long output_shape_1d[1];
      output_shape_1d[0] = 1;
      if(!OnnxSetOutputShape(handle, 0, output_shape_1d))
        {
         Print("Failed to set ONNX vector output shape for ", file_name, ". Err: ", GetLastError());
         return false;
        }
     }
   else
     {
      long output_shape_2d[2];
      output_shape_2d[0] = 1;
      output_shape_2d[1] = 1;
      if(!OnnxSetOutputShape(handle, 0, output_shape_2d))
        {
         Print("Failed to set ONNX matrix output shape for ", file_name, ". Err: ", GetLastError());
         return false;
        }
     }

   Print("ONNX model initialized: ", file_name);
   return true;
  }

//+------------------------------------------------------------------+
bool InitOnnx()
  {
   ArrayInitialize(ensemble_handles, INVALID_HANDLE);
   for(int i = 0; i < ArraySize(ensemble_handles); i++)
     {
      string file_name = gAssetFolder + _Symbol + "_" + ResolveTimeframeString() + "_" + ensemble_model_names[i] + ".onnx";
      if(!InitSingleOnnx(file_name, ensemble_model_names[i], ensemble_handles[i]))
         return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
bool LoadModelThresholds()
  {
   gActiveThresholdBuy = InpThresholdBuy;
   gActiveThresholdSell = InpThresholdSell;

   if(!InpUseModelThresholds)
      return true;

   int handle = OpenCommonCsv(gThresholdFileName, FILE_READ | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("Threshold file not found, using manual thresholds: ", gThresholdFileName);
      return true;
     }

   if(FileIsEnding(handle))
     {
      FileClose(handle);
      Print("Threshold file is empty, using manual thresholds: ", gThresholdFileName);
      return true;
     }

   string header_model = FileReadString(handle);
   string header_buy = FileReadString(handle);
   string header_sell = FileReadString(handle);
   if(header_model != "MODEL_NAME" || header_buy != "BUY_THRESHOLD" || header_sell != "SELL_THRESHOLD")
     {
      FileClose(handle);
      Print("Unexpected threshold header in ", gThresholdFileName, ", using manual thresholds.");
      return true;
     }

   while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
      FileReadString(handle);

   if(FileIsEnding(handle))
     {
      FileClose(handle);
      Print("Threshold file has no data rows, using manual thresholds: ", gThresholdFileName);
      return true;
     }

   string model_name = FileReadString(handle);
   string buy_value = FileReadString(handle);
   string sell_value = FileReadString(handle);
   FileClose(handle);

   if(model_name != ResolveModelString())
     {
      Print("Threshold file model mismatch, using manual thresholds: ", gThresholdFileName);
      return true;
     }

   gActiveThresholdBuy = StringToDouble(buy_value);
   gActiveThresholdSell = StringToDouble(sell_value);
   Print("Loaded model thresholds from ", gThresholdFileName,
         " | buy=", DoubleToString(gActiveThresholdBuy, 6),
         " | sell=", DoubleToString(gActiveThresholdSell, 6));
   return true;
  }

//+------------------------------------------------------------------+
void ReleaseModels()
  {
   if(onnx_handle != INVALID_HANDLE && onnx_handle > 0)
     {
      OnnxRelease(onnx_handle);
      onnx_handle = INVALID_HANDLE;
     }

   for(int i = 0; i < ArraySize(ensemble_handles); i++)
     {
      if(ensemble_handles[i] != INVALID_HANDLE && ensemble_handles[i] > 0)
        {
         OnnxRelease(ensemble_handles[i]);
         ensemble_handles[i] = INVALID_HANDLE;
        }
     }

   long indicator_handles[] = {
      gRsiHandle,
      gAtrHandle,
      gMacdHandle,
      gBandsHandle,
      gStochHandle,
      gAdxHandle,
      gRsiH4Handle,
      gMaH4Handle,
      gRsiM15Handle,
      gAtrM15Handle
   };
   for(int i = 0; i < ArraySize(indicator_handles); i++)
     {
      if(indicator_handles[i] != INVALID_HANDLE && indicator_handles[i] > 0)
         IndicatorRelease((int)indicator_handles[i]);
     }

   gRsiHandle = INVALID_HANDLE;
   gAtrHandle = INVALID_HANDLE;
   gMacdHandle = INVALID_HANDLE;
   gBandsHandle = INVALID_HANDLE;
   gStochHandle = INVALID_HANDLE;
   gAdxHandle = INVALID_HANDLE;
   gRsiH4Handle = INVALID_HANDLE;
   gMaH4Handle = INVALID_HANDLE;
   gRsiM15Handle = INVALID_HANDLE;
   gAtrM15Handle = INVALID_HANDLE;
  }

//+------------------------------------------------------------------+
bool InitIndicators()
  {
   gRsiHandle = iRSI(_Symbol, PERIOD_H1, 14, PRICE_CLOSE);
   gAtrHandle = iATR(_Symbol, PERIOD_H1, 14);
   gMacdHandle = iMACD(_Symbol, PERIOD_H1, 12, 26, 9, PRICE_CLOSE);
   gBandsHandle = iBands(_Symbol, PERIOD_H1, 20, 0, 2.0, PRICE_CLOSE);
   gStochHandle = iStochastic(_Symbol, PERIOD_H1, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
   gAdxHandle = iADX(_Symbol, PERIOD_H1, 14);
   gRsiH4Handle = iRSI(_Symbol, PERIOD_H4, 14, PRICE_CLOSE);
   gMaH4Handle = iMA(_Symbol, PERIOD_H4, 20, 0, MODE_SMA, PRICE_CLOSE);
   gRsiM15Handle = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
   gAtrM15Handle = iATR(_Symbol, PERIOD_M15, 14);

   if(gRsiHandle == INVALID_HANDLE || gAtrHandle == INVALID_HANDLE || gMacdHandle == INVALID_HANDLE ||
      gBandsHandle == INVALID_HANDLE || gStochHandle == INVALID_HANDLE || gAdxHandle == INVALID_HANDLE ||
      gRsiH4Handle == INVALID_HANDLE || gMaH4Handle == INVALID_HANDLE || gRsiM15Handle == INVALID_HANDLE ||
      gAtrM15Handle == INVALID_HANDLE)
     {
      Print("Failed to initialize one or more indicator handles. Err: ", GetLastError());
      return false;
     }
   return true;
  }

//+------------------------------------------------------------------+
bool LoadScalerParams()
  {
   int handle = OpenCommonCsv(gScalerFileName, FILE_READ | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open scaler file: ", gScalerFileName, " Err: ", GetLastError());
      return false;
     }

   ArrayResize(gFeatureNames, 0);
   ArrayResize(gScalerMeans, 0);
   ArrayResize(gScalerScales, 0);

   while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
      FileReadString(handle);

   int count = 0;
   while(!FileIsEnding(handle))
     {
      string feature_name = FileReadString(handle);
      if(StringLen(feature_name) == 0 && FileIsEnding(handle))
         break;

      string mean_value = FileReadString(handle);
      string scale_value = FileReadString(handle);

      ArrayResize(gFeatureNames, count + 1);
      ArrayResize(gScalerMeans, count + 1);
      ArrayResize(gScalerScales, count + 1);
      gFeatureNames[count] = feature_name;
      gScalerMeans[count] = StringToDouble(mean_value);
      gScalerScales[count] = StringToDouble(scale_value);
      if(gScalerScales[count] == 0.0)
         gScalerScales[count] = 1.0;
      count++;

      while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
         FileReadString(handle);
     }

   FileClose(handle);
   gFeatureCount = count;
   Print("Loaded ", count, " scaler features from ", gScalerFileName);
   return (gFeatureCount > 0);
  }

//+------------------------------------------------------------------+
bool IsRelevantCountryCode(const string country_code)
  {
   string base = StringSubstr(_Symbol, 0, 3);
   string quote = StringSubstr(_Symbol, 3, 3);
   string code = country_code;
   StringToUpper(code);

   if(base == "EUR" && code == "EU") return true;
   if(base == "USD" && code == "US") return true;
   if(base == "GBP" && code == "GB") return true;
   if(base == "AUD" && code == "AU") return true;
   if(base == "NZD" && code == "NZ") return true;
   if(base == "JPY" && code == "JP") return true;

   if(quote == "EUR" && code == "EU") return true;
   if(quote == "USD" && code == "US") return true;
   if(quote == "GBP" && code == "GB") return true;
   if(quote == "AUD" && code == "AU") return true;
   if(quote == "NZD" && code == "NZ") return true;
   if(quote == "JPY" && code == "JP") return true;

   return false;
  }

//+------------------------------------------------------------------+
string ResolveCountryCode(const string currency_code)
  {
   string cc = currency_code;
   StringToUpper(cc);
   if(cc == "EUR") return "EU";
   if(cc == "USD") return "US";
   if(cc == "GBP") return "GB";
   if(cc == "AUD") return "AU";
   if(cc == "NZD") return "NZ";
   if(cc == "JPY") return "JP";
   return "";
  }

//+------------------------------------------------------------------+
bool LoadNewsEvents()
  {
   int handle = OpenCommonCsv(gNewsEventsFileName, FILE_READ | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open news file: ", gNewsEventsFileName, " Err: ", GetLastError());
      return false;
     }

   ArrayResize(gNewsEventTimes, 0);
   ArrayResize(gNewsEventNames, 0);
   ArrayResize(gNewsCountryCodes, 0);
   ArrayResize(gNewsImportances, 0);
   ArrayResize(gNewsActuals, 0);
   ArrayResize(gNewsForecasts, 0);
   ArrayResize(gNewsPreviouss, 0);

   if(FileIsEnding(handle))
     {
      FileClose(handle);
      Print("NewsEvents.csv is empty.");
      return false;
     }

   while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
      FileReadString(handle);

   int count = 0;
   while(!FileIsEnding(handle))
     {
      string id_value = FileReadString(handle);
      if(StringLen(id_value) == 0 && FileIsEnding(handle))
         break;

      string datetime_value = FileReadString(handle);
      string time_value = FileReadString(handle);
      string name_value = FileReadString(handle);
      string country_code = FileReadString(handle);
      string country_name = FileReadString(handle);
      string importance_value = FileReadString(handle);
      string actual_value = FileReadString(handle);
      string forecast_value = FileReadString(handle);
      string previous_value = FileReadString(handle);
      string impact_value = FileReadString(handle);
      string url_value = FileReadString(handle);

      datetime event_time = StringToTime(datetime_value);
      int importance = (int)StringToInteger(importance_value);
      string normalized_code = country_code;
      StringToUpper(normalized_code);
      StringTrimLeft(normalized_code);
      StringTrimRight(normalized_code);

      if(event_time > 0)
        {
         ArrayResize(gNewsEventTimes, count + 1);
         ArrayResize(gNewsEventNames, count + 1);
         ArrayResize(gNewsCountryCodes, count + 1);
         ArrayResize(gNewsImportances, count + 1);
         gNewsEventTimes[count] = event_time;
         gNewsEventNames[count] = SanitizeCsvText(name_value);
         gNewsCountryCodes[count] = normalized_code;
         gNewsImportances[count] = importance;
         ArrayResize(gNewsActuals, count + 1);
         ArrayResize(gNewsForecasts, count + 1);
         ArrayResize(gNewsPreviouss, count + 1);
         gNewsActuals[count] = StringToDouble(actual_value);
         gNewsForecasts[count] = StringToDouble(forecast_value);
         gNewsPreviouss[count] = StringToDouble(previous_value);
         count++;
        }

      while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
         FileReadString(handle);
     }

   FileClose(handle);
   gLastNewsReloadTime = TimeCurrent();
   gNewsLoaded = true;
   if(count > 0)
      Print("Loaded ", count, " total rows from ", gNewsEventsFileName);
   else
      Print("Loaded 0 rows from ", gNewsEventsFileName, ". EA will stay active and wait for future news reloads.");
   return true;
  }

//+------------------------------------------------------------------+
bool RefreshNewsEventsIfDue(const bool force_reload=false)
  {
   datetime now = TimeCurrent();
   if(!force_reload && gNewsLoaded && (now - gLastNewsReloadTime) < (long)InpNewsReloadSeconds)
      return true;
   return LoadNewsEvents();
  }

//+------------------------------------------------------------------+
bool ReadIndicatorValue(const long handle, const int buffer_index, const int shift, double &value)
  {
   double buffer[];
   ArraySetAsSeries(buffer, true);
   if(CopyBuffer((int)handle, buffer_index, shift, 1, buffer) != 1)
      return false;
   value = buffer[0];
   return true;
  }

//+------------------------------------------------------------------+
double ComputeVolatility20(MqlRates &rates[], const int shift)
  {
   double returns[20];
   double sum = 0.0;
   for(int i = 0; i < 20; i++)
     {
      double close_now = rates[shift + i].close;
      double close_prev = rates[shift + i + 1].close;
      if(close_now <= 0.0 || close_prev <= 0.0)
         return 0.0;
      returns[i] = MathLog(close_now / close_prev);
      sum += returns[i];
     }

   double mean = sum / 20.0;
   double variance = 0.0;
   for(int i = 0; i < 20; i++)
     {
      double diff = returns[i] - mean;
      variance += diff * diff;
     }
   return MathSqrt(variance / 20.0);
  }

//+------------------------------------------------------------------+
bool BuildScaledFeatureVector(const datetime bar_time, vectorf &features, int &max_importance, string &event_name)
  {
   event_name = "Economic Event";
   max_importance = 0;

   int shift = iBarShift(_Symbol, PERIOD_H1, bar_time, true);
   if(shift < 0)
      return false;

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   if(CopyRates(_Symbol, PERIOD_H1, 0, shift + 25, rates) < (shift + 21))
      return false;

   string base_code = ResolveCountryCode(StringSubstr(_Symbol, 0, 3));
   string quote_code = ResolveCountryCode(StringSubstr(_Symbol, 3, 3));
   double news_event_count = 0.0;
   double news_max_importance = 0.0;
   double news_nearest_offset_hours = 0.0;
   double base_news_count = 0.0;
   double base_max_importance = 0.0;
   double base_actual_sum = 0.0;
   double base_forecast_sum = 0.0;
   double base_previous_sum = 0.0;
   double base_surprise_af_sum = 0.0;
   double base_surprise_ap_sum = 0.0;
   double quote_news_count = 0.0;
   double quote_max_importance = 0.0;
   double quote_actual_sum = 0.0;
   double quote_forecast_sum = 0.0;
   double quote_previous_sum = 0.0;
   double quote_surprise_af_sum = 0.0;
   double quote_surprise_ap_sum = 0.0;
   double best_abs_offset = DBL_MAX;
   int best_importance = -1;
   bool has_news = false;

   for(int i = 0; i < ArraySize(gNewsEventTimes); i++)
     {
      datetime event_time = gNewsEventTimes[i];
      string event_country = gNewsCountryCodes[i];
      if(event_country != base_code && event_country != quote_code)
         continue;

      if(event_time > bar_time)
         continue;

      int diff_seconds = (int)(bar_time - event_time);
      if(diff_seconds < 0 || diff_seconds > 7200)
         continue;

      has_news = true;
      double offset_hours = (double)diff_seconds / 3600.0;
      double importance = (double)gNewsImportances[i];
      news_event_count += 1.0;
      if(importance > news_max_importance)
         news_max_importance = importance;
      if(offset_hours < best_abs_offset || (MathAbs(offset_hours - best_abs_offset) < 0.000001 && gNewsImportances[i] > best_importance))
        {
         best_abs_offset = offset_hours;
         best_importance = gNewsImportances[i];
         news_nearest_offset_hours = offset_hours;
         event_name = gNewsEventNames[i];
        }

      double actual = gNewsActuals[i];
      double forecast = gNewsForecasts[i];
      double previous = gNewsPreviouss[i];
      double surprise_af = actual - forecast;
      double surprise_ap = actual - previous;

       if(event_country == base_code)
         {
         base_news_count += 1.0;
         if(importance > base_max_importance)
            base_max_importance = importance;
         base_actual_sum += actual;
         base_forecast_sum += forecast;
         base_previous_sum += previous;
         base_surprise_af_sum += surprise_af;
         base_surprise_ap_sum += surprise_ap;
        }
       else if(event_country == quote_code)
         {
         quote_news_count += 1.0;
         if(importance > quote_max_importance)
            quote_max_importance = importance;
         quote_actual_sum += actual;
         quote_forecast_sum += forecast;
         quote_previous_sum += previous;
         quote_surprise_af_sum += surprise_af;
         quote_surprise_ap_sum += surprise_ap;
        }
     }

   if(!has_news)
      return false;

   max_importance = (int)news_max_importance;

   double rsi = 0.0, atr = 0.0, macd_main = 0.0, macd_signal = 0.0;
   double bands_upper = 0.0, bands_lower = 0.0, stoch_main = 0.0, stoch_signal = 0.0;
   double adx_main = 0.0, adx_plus_di = 0.0, adx_minus_di = 0.0;
   double rsi_h4 = 0.0, ma_h4 = 0.0, rsi_m15 = 0.0, atr_m15 = 0.0;

   if(!ReadIndicatorValue(gRsiHandle, 0, shift, rsi)) return false;
   if(!ReadIndicatorValue(gAtrHandle, 0, shift, atr)) return false;
   if(!ReadIndicatorValue(gMacdHandle, 0, shift, macd_main)) return false;
   if(!ReadIndicatorValue(gMacdHandle, 1, shift, macd_signal)) return false;
   if(!ReadIndicatorValue(gBandsHandle, 1, shift, bands_upper)) return false;
   if(!ReadIndicatorValue(gBandsHandle, 2, shift, bands_lower)) return false;
   if(!ReadIndicatorValue(gStochHandle, 0, shift, stoch_main)) return false;
   if(!ReadIndicatorValue(gStochHandle, 1, shift, stoch_signal)) return false;
   if(!ReadIndicatorValue(gAdxHandle, 0, shift, adx_main)) return false;
   if(!ReadIndicatorValue(gAdxHandle, 1, shift, adx_plus_di)) return false;
   if(!ReadIndicatorValue(gAdxHandle, 2, shift, adx_minus_di)) return false;

   int shift_h4 = iBarShift(_Symbol, PERIOD_H4, bar_time, false);
   int shift_m15 = iBarShift(_Symbol, PERIOD_M15, bar_time, false);
   if(shift_h4 < 0 || shift_m15 < 0)
      return false;
   if(!ReadIndicatorValue(gRsiH4Handle, 0, shift_h4, rsi_h4)) return false;
   if(!ReadIndicatorValue(gMaH4Handle, 0, shift_h4, ma_h4)) return false;
   if(!ReadIndicatorValue(gRsiM15Handle, 0, shift_m15, rsi_m15)) return false;
   if(!ReadIndicatorValue(gAtrM15Handle, 0, shift_m15, atr_m15)) return false;

   double open = rates[shift].open;
   double high = rates[shift].high;
   double low = rates[shift].low;
   double close = rates[shift].close;
   double tick_volume = (double)rates[shift].tick_volume;
   MqlDateTime bar_struct;
   TimeToStruct(bar_time, bar_struct);
   double hour = (double)bar_struct.hour;
   double day_of_week = (double)bar_struct.day_of_week;
   double body = close - open;
   double upper_wick = high - MathMax(open, close);
   double lower_wick = MathMin(open, close) - low;
   double log_return = 0.0;
   if(rates[shift + 1].close > 0.0 && close > 0.0)
      log_return = MathLog(close / rates[shift + 1].close);
   double volatility_20 = ComputeVolatility20(rates, shift);

   features.Resize(gFeatureCount);
   for(int i = 0; i < gFeatureCount; i++)
     {
      string feature = gFeatureNames[i];
      double raw_value = 0.0;

      if(feature == "open") raw_value = open;
      else if(feature == "high") raw_value = high;
      else if(feature == "low") raw_value = low;
      else if(feature == "close") raw_value = close;
      else if(feature == "tick_volume") raw_value = tick_volume;
      else if(feature == "hour") raw_value = hour;
      else if(feature == "day_of_week") raw_value = day_of_week;
      else if(feature == "rsi") raw_value = rsi;
      else if(feature == "atr") raw_value = atr;
      else if(feature == "macd_main") raw_value = macd_main;
      else if(feature == "macd_signal") raw_value = macd_signal;
      else if(feature == "bands_upper") raw_value = bands_upper;
      else if(feature == "bands_lower") raw_value = bands_lower;
      else if(feature == "stoch_main") raw_value = stoch_main;
      else if(feature == "stoch_signal") raw_value = stoch_signal;
      else if(feature == "adx_main") raw_value = adx_main;
      else if(feature == "adx_plus_di") raw_value = adx_plus_di;
      else if(feature == "adx_minus_di") raw_value = adx_minus_di;
      else if(feature == "rsi_h4") raw_value = rsi_h4;
      else if(feature == "ma_h4") raw_value = ma_h4;
      else if(feature == "rsi_m15") raw_value = rsi_m15;
      else if(feature == "atr_m15") raw_value = atr_m15;
      else if(feature == "body") raw_value = body;
      else if(feature == "upper_wick") raw_value = upper_wick;
      else if(feature == "lower_wick") raw_value = lower_wick;
      else if(feature == "log_return") raw_value = log_return;
      else if(feature == "volatility_20") raw_value = volatility_20;
      else if(feature == "news_event_present") raw_value = 1.0;
      else if(feature == "news_event_count") raw_value = news_event_count;
      else if(feature == "news_max_importance") raw_value = news_max_importance;
      else if(feature == "news_nearest_offset_hours") raw_value = news_nearest_offset_hours;
      else if(feature == "base_news_count") raw_value = base_news_count;
      else if(feature == "base_max_importance") raw_value = base_max_importance;
      else if(feature == "base_actual_sum") raw_value = base_actual_sum;
      else if(feature == "base_forecast_sum") raw_value = base_forecast_sum;
      else if(feature == "base_previous_sum") raw_value = base_previous_sum;
      else if(feature == "base_surprise_actual_forecast_sum") raw_value = base_surprise_af_sum;
      else if(feature == "base_surprise_actual_previous_sum") raw_value = base_surprise_ap_sum;
      else if(feature == "quote_news_count") raw_value = quote_news_count;
      else if(feature == "quote_max_importance") raw_value = quote_max_importance;
      else if(feature == "quote_actual_sum") raw_value = quote_actual_sum;
      else if(feature == "quote_forecast_sum") raw_value = quote_forecast_sum;
      else if(feature == "quote_previous_sum") raw_value = quote_previous_sum;
      else if(feature == "quote_surprise_actual_forecast_sum") raw_value = quote_surprise_af_sum;
      else if(feature == "quote_surprise_actual_previous_sum") raw_value = quote_surprise_ap_sum;

      features[i] = (float)((raw_value - gScalerMeans[i]) / gScalerScales[i]);
     }

   return true;
  }

//+------------------------------------------------------------------+
bool RunPredictionForHandle(const long handle, const string model_name, const vectorf &features, double &prediction)
  {
   vectorf feature_copy = features;
   vectorf prediction_output(1);
   if(!OnnxRun(handle, ONNX_DEFAULT, feature_copy, prediction_output))
     {
      Print("ONNX inference failed for ", model_name, ". Err: ", GetLastError());
      return false;
     }

   prediction = prediction_output[0];
   return true;
  }

//+------------------------------------------------------------------+
bool EvaluateEvent(const datetime bar_time, double &prediction, int &importance, string &event_name)
  {
   vectorf features;
   if(!BuildScaledFeatureVector(bar_time, features, importance, event_name))
      return false;

   double prediction_sum = 0.0;
   int valid_models = 0;
   for(int i = 0; i < ArraySize(ensemble_handles); i++)
     {
      double model_prediction = 0.0;
      if(!RunPredictionForHandle(ensemble_handles[i], ensemble_model_names[i], features, model_prediction))
         return false;
      prediction_sum += model_prediction;
      valid_models++;
     }

   if(valid_models <= 0)
      return false;

   prediction = prediction_sum / valid_models;
   event_name = SanitizeCsvText(event_name);
   if(StringLen(event_name) == 0)
      event_name = "Economic Event";
   return true;
  }

//+------------------------------------------------------------------+
string SanitizeCsvText(const string value)
  {
   string sanitized = value;
   StringReplace(sanitized, "\r", " ");
   StringReplace(sanitized, "\n", " ");
   StringReplace(sanitized, "\"", "");
   StringReplace(sanitized, ",", ";");
   StringTrimLeft(sanitized);
   StringTrimRight(sanitized);
   return sanitized;
  }

//+------------------------------------------------------------------+
bool AppendDebugCsv(const datetime previous_bar_time,
                    const datetime current_bar_time,
                    const string event_name,
                    const int importance,
                    const double prediction,
                    const int signal,
                    const int current_spread,
                    const bool executed,
                    const string status)
  {
   if(!InpEnableDebugCsv)
      return true;

   if(StringLen(gDebugCsvFileName) == 0)
      gDebugCsvFileName = InpDebugCsvFileName;

   if(StringLen(gDebugCsvFileName) == 0)
      gDebugCsvFileName = "Regression_EA_debug.csv";

   bool file_exists = FileIsExist(gDebugCsvFileName, FILE_COMMON);
   int handle = OpenCommonCsv(gDebugCsvFileName, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open debug CSV: ", gDebugCsvFileName, " Err: ", GetLastError());
      return false;
     }

   if(!file_exists || FileSize(handle) <= 0)
     {
      FileWrite(handle,
                "logged_at",
                "bar_time",
                "current_bar_time",
                "symbol",
                "event_name",
                "importance",
                "prediction",
                "buy_threshold",
                "sell_threshold",
                "signal",
                "spread",
                "executed",
                "status",
                "news_rows",
                "last_news_reload");
     }
   else
     {
      FileSeek(handle, 0, SEEK_END);
     }

   FileWrite(handle,
             TimeToString(TimeCurrent(), TIME_DATE | TIME_SECONDS),
             TimeToString(previous_bar_time, TIME_DATE | TIME_MINUTES),
             TimeToString(current_bar_time, TIME_DATE | TIME_MINUTES),
             _Symbol,
             SanitizeCsvText(event_name),
             IntegerToString(importance),
             DoubleToString(prediction, 6),
             DoubleToString(gActiveThresholdBuy, 6),
             DoubleToString(gActiveThresholdSell, 6),
             IntegerToString(signal),
             IntegerToString(current_spread),
             (executed ? "1" : "0"),
             SanitizeCsvText(status),
             IntegerToString(ArraySize(gNewsEventTimes)),
             (gLastNewsReloadTime > 0 ? TimeToString(gLastNewsReloadTime, TIME_DATE | TIME_SECONDS) : ""));
   FileClose(handle);
   return true;
  }

//+------------------------------------------------------------------+
double CalculateOrderLotSize()
  {
   double min_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volume_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);

   if(min_volume <= 0.0)
      min_volume = 0.01;
   if(max_volume < min_volume)
      max_volume = min_volume;
   if(volume_step <= 0.0)
      volume_step = min_volume;

   if(InpStopLoss <= 0 || point <= 0.0 || tick_size <= 0.0 || tick_value <= 0.0 || balance <= 0.0 || InpRiskPercent <= 0.0)
      return min_volume;

   double risk_amount = balance * (InpRiskPercent / 100.0);
   double stop_distance = InpStopLoss * point;
   double ticks_to_stop = stop_distance / tick_size;
   double risk_per_lot = ticks_to_stop * tick_value;

   if(risk_per_lot <= 0.0)
      return min_volume;

   double raw_lot = risk_amount / risk_per_lot;
   double steps = MathFloor(raw_lot / volume_step);
   double normalized_lot = steps * volume_step;

   if(normalized_lot < min_volume)
      normalized_lot = min_volume;
   if(normalized_lot > max_volume)
      normalized_lot = max_volume;

   int volume_digits = 2;
   if(volume_step > 0.0)
      volume_digits = (int)MathMax(0.0, MathRound(-MathLog10(volume_step)));

   return NormalizeDouble(normalized_lot, volume_digits);
  }

//+------------------------------------------------------------------+
int CountOpenPositions(int &buy_count, int &sell_count)
  {
   buy_count = 0;
   sell_count = 0;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;

      long position_type = PositionGetInteger(POSITION_TYPE);
      if(position_type == POSITION_TYPE_BUY)
         buy_count++;
      else if(position_type == POSITION_TYPE_SELL)
         sell_count++;
     }

   return buy_count + sell_count;
  }

//+------------------------------------------------------------------+
void ClosePositions(ENUM_POSITION_TYPE type)
  {
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if((ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if(PositionGetInteger(POSITION_TYPE) == type)
         trade.PositionClose(ticket);
     }
  }

//+------------------------------------------------------------------+
bool ExecuteSignal(const int signal, const double prediction, const string order_comment)
  {
   if(signal == 0)
      return false;

   int buy_count = 0;
   int sell_count = 0;
   int total_positions = CountOpenPositions(buy_count, sell_count);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double margin_free = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
   double lot_size = CalculateOrderLotSize();

   DebugPrint(StringFormat(
      "Signal execution check | signal=%d | prediction=%.6f | lot=%.2f | balance=%.2f | equity=%.2f | free_margin=%.2f | open_positions=%d",
      signal,
      prediction,
      lot_size,
      balance,
      equity,
      margin_free,
      total_positions
   ));

   if(signal > 0)
     {
      if(sell_count > 0 && InpAllowReverse)
        {
         ClosePositions(POSITION_TYPE_SELL);
         total_positions = CountOpenPositions(buy_count, sell_count);
        }

      if(total_positions < InpMaxOpenPosition && buy_count == 0)
        {
         double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double sl = (InpStopLoss > 0 ? ask - InpStopLoss * point : 0);
         double tp = (InpTakeProfit > 0 ? ask + InpTakeProfit * point : 0);
         DebugPrint(StringFormat(
            "Attempt BUY | ask=%.5f | sl=%.5f | tp=%.5f | comment=%s",
            ask,
            sl,
            tp,
            order_comment
         ));
         if(trade.Buy(lot_size, _Symbol, ask, sl, tp, order_comment))
           {
            Print("BUY Executed. Pred: ", prediction, " | Event: ", order_comment);
            return true;
           }
         Print("BUY FAILED! Error Code: ", trade.ResultRetcode(), " | Result: ", trade.ResultComment());
        }
      return false;
     }

   if(buy_count > 0 && InpAllowReverse)
     {
      ClosePositions(POSITION_TYPE_BUY);
      total_positions = CountOpenPositions(buy_count, sell_count);
     }

   if(total_positions < InpMaxOpenPosition && sell_count == 0)
     {
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double sl = (InpStopLoss > 0 ? bid + InpStopLoss * point : 0);
      double tp = (InpTakeProfit > 0 ? bid - InpTakeProfit * point : 0);
      DebugPrint(StringFormat(
         "Attempt SELL | bid=%.5f | sl=%.5f | tp=%.5f | comment=%s",
         bid,
         sl,
         tp,
         order_comment
      ));
      if(trade.Sell(lot_size, _Symbol, bid, sl, tp, order_comment))
        {
         Print("SELL Executed. Pred: ", prediction, " | Event: ", order_comment);
         return true;
        }
      Print("SELL FAILED! Error Code: ", trade.ResultRetcode(), " | Result: ", trade.ResultComment());
     }
   return false;
  }

//+------------------------------------------------------------------+
bool ProcessRuntimeWindow(const datetime previous_bar_time, const datetime current_bar_time, const int current_spread)
  {
   int min_importance = ResolveMinimumImportance();
   double prediction = 0.0;
   int importance = 0;
   string event_name = "";
   if(!EvaluateEvent(previous_bar_time, prediction, importance, event_name))
     {
      AppendDebugCsv(previous_bar_time, current_bar_time, "", 0, 0.0, 0, current_spread, false, "no_news_or_feature_data");
      return false;
     }

   if(InpTraceEventWindows)
      DebugPrint(StringFormat(
         "Live event evaluated | previous=%s | current=%s | event=%s | importance=%d | prediction=%.6f",
         TimeToString(previous_bar_time, TIME_DATE | TIME_MINUTES),
         TimeToString(current_bar_time, TIME_DATE | TIME_MINUTES),
         event_name,
         importance,
         prediction
      ));

   if(importance < min_importance)
     {
      AppendDebugCsv(previous_bar_time, current_bar_time, event_name, importance, prediction, 0, current_spread, false, "below_importance_filter");
      return false;
     }

   int signal = 0;
   if(prediction >= gActiveThresholdBuy)
      signal = 1;
   else if(prediction <= gActiveThresholdSell)
      signal = -1;
   if(signal == 0)
     {
      AppendDebugCsv(previous_bar_time, current_bar_time, event_name, importance, prediction, 0, current_spread, false, "below_threshold");
      return false;
     }

   if(InpMaxSpread > 0 && current_spread > InpMaxSpread)
     {
      DebugPrint(StringFormat(
         "Spread filter blocked signal | bar=%s | event=%s | spread=%d | max=%d | prediction=%.6f | signal=%d",
         TimeToString(previous_bar_time, TIME_DATE | TIME_MINUTES),
         event_name,
         current_spread,
         InpMaxSpread,
         prediction,
         signal
      ));
      AppendDebugCsv(previous_bar_time, current_bar_time, event_name, importance, prediction, signal, current_spread, false, "spread_blocked");
      return false;
     }

   DebugPrint(StringFormat(
      "Best signal selected | bar=%s | event=%s | prediction=%.6f | signal=%d",
      TimeToString(previous_bar_time, TIME_DATE | TIME_MINUTES),
      event_name,
      prediction,
      signal
   ));

   bool executed = ExecuteSignal(signal, prediction, event_name);
   AppendDebugCsv(previous_bar_time, current_bar_time, event_name, importance, prediction, signal, current_spread, executed, (executed ? "trade_executed" : "trade_failed_or_skipped"));
   return executed;
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   onnx_handle = INVALID_HANDLE;
   ArrayInitialize(ensemble_handles, INVALID_HANDLE);

   if(_Period != PERIOD_H1)
     {
      Print("Regression_EA supports H1 only. Current period: ", EnumToString(_Period));
      return INIT_FAILED;
     }

   trade.SetExpertMagicNumber(InpMagicNumber);

   gThresholdFileName = ResolveThresholdFileName();
   gScalerFileName = ResolveScalerFileName();
   gDebugCsvFileName = InpDebugCsvFileName;

   if(!LoadScalerParams())
      return INIT_FAILED;
   if(!InitIndicators())
      return INIT_FAILED;
   if(!RefreshNewsEventsIfDue(true))
      return INIT_FAILED;
   if(!LoadModelThresholds())
      return INIT_FAILED;
   if(!InitOnnx())
      return INIT_FAILED;

   Print("Regression_EA initialized successfully. Symbol: ", _Symbol,
         " | Model: ensemble_regression",
         " | Scaler: ", gScalerFileName,
         " | News file: ", gNewsEventsFileName,
         " | Debug CSV: ", gDebugCsvFileName,
         " | NewsFilterMinImportance: ", ResolveMinimumImportance(),
         " | ThresholdBuy: ", DoubleToString(gActiveThresholdBuy, 6),
         " | ThresholdSell: ", DoubleToString(gActiveThresholdSell, 6),
         " | MaxSpread: ", InpMaxSpread);
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ReleaseModels();
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   RefreshNewsEventsIfDue(false);

   if(InpTradeOnlyNewBar && !IsNewBar())
      return;

   long spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);

   MqlRates rates[2];
   if(CopyRates(_Symbol, PERIOD_H1, 0, 2, rates) < 2)
      return;

   datetime first_rate_time = rates[0].time;
   datetime second_rate_time = rates[1].time;
   datetime current_bar_time = (first_rate_time > second_rate_time ? first_rate_time : second_rate_time);
   datetime previous_bar_time = (first_rate_time > second_rate_time ? second_rate_time : first_rate_time);

   if(InpEnableDebugLog && !gLoggedRatesOrder)
     {
      Print(StringFormat(
         "CopyRates order check | rates0=%s | rates1=%s | previous=%s | current=%s",
         TimeToString(first_rate_time, TIME_DATE | TIME_MINUTES),
         TimeToString(second_rate_time, TIME_DATE | TIME_MINUTES),
         TimeToString(previous_bar_time, TIME_DATE | TIME_MINUTES),
         TimeToString(current_bar_time, TIME_DATE | TIME_MINUTES)
      ));
      gLoggedRatesOrder = true;
     }

   if(previous_bar_time >= current_bar_time)
      return;

   DebugRuntimeStatus(previous_bar_time, current_bar_time, (int)spread);
   ProcessRuntimeWindow(previous_bar_time, current_bar_time, (int)spread);
  }
//+------------------------------------------------------------------+
