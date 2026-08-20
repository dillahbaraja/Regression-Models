//+------------------------------------------------------------------+
//|                                         ML_Regression_EA.mq5     |
//|      News-driven ONNX regression trader for runtime event bars   |
//+------------------------------------------------------------------+
#property strict
#property copyright "Copyright 2026, Abdillah Baradja"
#property link      "dillahbaraja@gmail.com"
#property version   "3.00"

#include <Trade\Trade.mqh>

enum ENUM_MODEL_TYPE
  {
   MODEL_XGBOOST,
   MODEL_LIGHTGBM,
   MODEL_CATBOOST,
   MODEL_SKLEARN_HGB,
   MODEL_ENSEMBLE_REGRESSION
  };

enum ENUM_NEWS_FILTER_MODE
  {
   NEWS_FILTER_ALL,
   NEWS_FILTER_MEDIUM_HIGH,
   NEWS_FILTER_HIGH
  };

enum ENUM_LOT_MODE
  {
   LOT_MODE_STATIC,
   LOT_MODE_DYNAMIC_RISK
  };

input ENUM_MODEL_TYPE InpModelType             = MODEL_XGBOOST;
input ENUM_NEWS_FILTER_MODE InpNewsFilterMode  = NEWS_FILTER_ALL;
input ENUM_LOT_MODE  InpLotMode                = LOT_MODE_STATIC;
input double          InpLotSize               = 0.1;
input double          InpRiskPercent           = 5.0;
input ulong           InpMagicNumber           = 123456;
input int             InpMaxSpread             = 20;
input int             InpStopLoss              = 200;
input int             InpTakeProfit            = 400;
input int             InpMaxOpenPosition       = 1;
input bool            InpAllowReverse          = true;
input bool            InpTradeOnlyNewBar       = true;
input bool            InpUseModelThresholds    = true;
input double          InpThresholdBuy          = 0.00015;
input double          InpThresholdSell         = -0.00015;
input bool            InpEnableTransactionCsv  = true;
input bool            InpExportHistoryOnDeinit = true;
input bool            InpEnableDebugLog        = true;
input bool            InpTraceEventWindows     = true;
input int             InpDebugStatusEveryBars  = 250;
input string          InpTransactionCsvFile    = "";

CTrade trade;

long onnx_handle = INVALID_HANDLE;
long ensemble_handles[4];
string ensemble_model_names[4] = {"xgboost", "lightgbm", "catboost", "sklearn_hgb"};

datetime gLastBarTime = 0;
string gRuntimeFileName = "";
string gRuntimeDetailFileName = "";
string gTransactionCsvFile = "";
string gThresholdFileName = "";
string gAssetFolder = "Regression_Assets\\";
int gFeatureCount = 0;
int gRuntimeEventCount = 0;
int gRuntimeEventCapacity = 0;
int gNextEventIndex = 0;
int gDebugBarCounter = 0;
bool gLoggedRatesOrder = false;
double gActiveThresholdBuy = 0.0;
double gActiveThresholdSell = 0.0;

datetime gRuntimeBarTimes[];
int gRuntimeImportances[];
double gRuntimeFeatures[];

datetime gDetailEventTimes[];
string gDetailEventNames[];
string gDetailCountryCodes[];
int gDetailImportances[];

bool IsNewBar();
bool IsEnsembleModelType();
bool IsVectorOutputModelName(const string model_name);
string ResolveModelString();
string ResolveTimeframeString();
string ResolveRuntimeFileName();
string ResolveRuntimeDetailFileName();
string ResolveTransactionCsvFile();
string ResolveThresholdFileName();
int OpenCommonCsv(const string file_name, const int flags);
int ResolveMinimumImportance();
bool InitSingleOnnx(const string file_name, const string model_name, long &handle);
bool InitOnnx();
bool LoadModelThresholds();
void ReleaseModels();
bool LoadRuntimeEvents();
void EnsureRuntimeCapacity(const int needed_count);
void SaveRuntimeFeature(const int event_index, const int feature_index, const double value);
double GetRuntimeFeature(const int event_index, const int feature_index);
bool RunPredictionForHandle(const long handle, const string model_name, const vectorf &features, double &prediction);
bool EvaluateEvent(const int event_index, double &prediction);
bool ProcessRuntimeWindow(const datetime previous_bar_time, const datetime current_bar_time, const int current_spread);
int CountOpenPositions(int &buy_count, int &sell_count);
void ClosePositions(ENUM_POSITION_TYPE type);
bool ExecuteSignal(const int signal, const double prediction, const string order_comment);
double CalculateOrderLotSize();
string ResolveEventComment(const int event_index);
string SanitizeCsvText(const string value);
string DealTypeToText(const ENUM_DEAL_TYPE deal_type);
string DealEntryToText(const ENUM_DEAL_ENTRY deal_entry);
bool AppendDealCsv(const ulong deal_ticket);
bool ExportAllDealHistoryCsv();
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

   string next_event_text = "none";
   if(gNextEventIndex >= 0 && gNextEventIndex < gRuntimeEventCount)
      next_event_text = TimeToString(gRuntimeBarTimes[gNextEventIndex], TIME_DATE | TIME_MINUTES);

   Print(StringFormat(
      "Runtime status | previous=%s | current=%s | spread=%d | next_event_index=%d/%d | next_event_bar=%s",
      TimeToString(previous_bar_time, TIME_DATE | TIME_MINUTES),
      TimeToString(current_bar_time, TIME_DATE | TIME_MINUTES),
      current_spread,
      gNextEventIndex,
      gRuntimeEventCount,
      next_event_text
   ));
  }

//+------------------------------------------------------------------+
bool IsEnsembleModelType()
  {
   return InpModelType == MODEL_ENSEMBLE_REGRESSION;
  }

//+------------------------------------------------------------------+
bool IsVectorOutputModelName(const string model_name)
  {
   return false;
  }

//+------------------------------------------------------------------+
string ResolveModelString()
  {
   if(InpModelType == MODEL_XGBOOST)
      return "xgboost";
   if(InpModelType == MODEL_LIGHTGBM)
      return "lightgbm";
   if(InpModelType == MODEL_CATBOOST)
      return "catboost";
   if(InpModelType == MODEL_SKLEARN_HGB)
      return "sklearn_hgb";
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
string ResolveRuntimeFileName()
  {
   return gAssetFolder + _Symbol + "_" + ResolveTimeframeString() + "_runtime_events.csv";
  }

//+------------------------------------------------------------------+
string ResolveRuntimeDetailFileName()
  {
   return gAssetFolder + _Symbol + "_" + ResolveTimeframeString() + "_runtime_event_details.csv";
  }

//+------------------------------------------------------------------+
string ResolveTransactionCsvFile()
  {
   if(StringLen(InpTransactionCsvFile) > 0)
      return InpTransactionCsvFile;
   return gAssetFolder + _Symbol + "_" + ResolveTimeframeString() + "_" + ResolveModelString() + "_transactions.csv";
  }

//+------------------------------------------------------------------+
string ResolveThresholdFileName()
  {
   return gAssetFolder + _Symbol + "_" + ResolveTimeframeString() + "_" + ResolveModelString() + "_thresholds.csv";
  }

//+------------------------------------------------------------------+
int OpenCommonCsv(const string file_name, const int flags)
  {
   return FileOpen(file_name, flags | FILE_COMMON, ',');
  }

//+------------------------------------------------------------------+
int ResolveMinimumImportance()
  {
   if(InpNewsFilterMode == NEWS_FILTER_HIGH)
      return 3;
   if(InpNewsFilterMode == NEWS_FILTER_MEDIUM_HIGH)
      return 2;
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

   if(IsEnsembleModelType())
     {
      for(int i = 0; i < ArraySize(ensemble_handles); i++)
        {
         string file_name = gAssetFolder + _Symbol + "_" + ResolveTimeframeString() + "_" + ensemble_model_names[i] + ".onnx";
         if(!InitSingleOnnx(file_name, ensemble_model_names[i], ensemble_handles[i]))
            return false;
        }
      return true;
     }

   string file_name = _Symbol + "_" + ResolveTimeframeString() + "_" + ResolveModelString() + ".onnx";
  return InitSingleOnnx(gAssetFolder + file_name, ResolveModelString(), onnx_handle);
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
  }

//+------------------------------------------------------------------+
void EnsureRuntimeCapacity(const int needed_count)
  {
   if(needed_count <= gRuntimeEventCapacity)
      return;

   int new_capacity = (gRuntimeEventCapacity == 0 ? 256 : gRuntimeEventCapacity * 2);
   while(new_capacity < needed_count)
      new_capacity *= 2;

   ArrayResize(gRuntimeBarTimes, new_capacity);
   ArrayResize(gRuntimeImportances, new_capacity);
   ArrayResize(gDetailEventTimes, new_capacity);
   ArrayResize(gDetailEventNames, new_capacity);
   ArrayResize(gDetailCountryCodes, new_capacity);
   ArrayResize(gDetailImportances, new_capacity);

   if(gFeatureCount > 0)
      ArrayResize(gRuntimeFeatures, new_capacity * gFeatureCount);

   gRuntimeEventCapacity = new_capacity;
  }

//+------------------------------------------------------------------+
void SaveRuntimeFeature(const int event_index, const int feature_index, const double value)
  {
   gRuntimeFeatures[event_index * gFeatureCount + feature_index] = value;
  }

//+------------------------------------------------------------------+
double GetRuntimeFeature(const int event_index, const int feature_index)
  {
   return gRuntimeFeatures[event_index * gFeatureCount + feature_index];
  }

//+------------------------------------------------------------------+
bool LoadRuntimeEvents()
  {
   int handle = OpenCommonCsv(gRuntimeFileName, FILE_READ | FILE_CSV | FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("Failed to open runtime event file: ", gRuntimeFileName, " Err: ", GetLastError());
      return false;
     }

   if(FileIsEnding(handle))
     {
      FileClose(handle);
      Print("Runtime event file is empty: ", gRuntimeFileName);
      return false;
     }

   string header_time = FileReadString(handle);
   string header_importance = FileReadString(handle);
   if(header_time != "BAR_TIME" || header_importance != "IMPORTANCE")
     {
      FileClose(handle);
      Print("Unexpected runtime header in ", gRuntimeFileName);
      return false;
     }

   gFeatureCount = 0;
   while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
     {
      FileReadString(handle);
      gFeatureCount++;
     }

   if(gFeatureCount <= 0)
     {
      FileClose(handle);
      Print("No runtime features found in ", gRuntimeFileName);
      return false;
     }

   gRuntimeEventCount = 0;
   gRuntimeEventCapacity = 0;
   ArrayResize(gRuntimeFeatures, 0);
   EnsureRuntimeCapacity(1);

   while(!FileIsEnding(handle))
     {
      string bar_time_str = FileReadString(handle);
      if(StringLen(bar_time_str) == 0 && FileIsEnding(handle))
         break;

      string importance_str = FileReadString(handle);
      datetime bar_time = StringToTime(bar_time_str);
      if(bar_time <= 0)
        {
         while(!FileIsLineEnding(handle) && !FileIsEnding(handle))
            FileReadString(handle);
         continue;
        }

      EnsureRuntimeCapacity(gRuntimeEventCount + 1);
      gRuntimeBarTimes[gRuntimeEventCount] = bar_time;
      gRuntimeImportances[gRuntimeEventCount] = (int)StringToInteger(importance_str);

      for(int i = 0; i < gFeatureCount; i++)
        {
         string raw_value = FileReadString(handle);
         SaveRuntimeFeature(gRuntimeEventCount, i, StringToDouble(raw_value));
        }

      gDetailEventTimes[gRuntimeEventCount] = 0;
      gDetailEventNames[gRuntimeEventCount] = "";
      gDetailCountryCodes[gRuntimeEventCount] = "";
      gDetailImportances[gRuntimeEventCount] = gRuntimeImportances[gRuntimeEventCount];

      gRuntimeEventCount++;
     }

   FileClose(handle);

   ArrayResize(gRuntimeBarTimes, gRuntimeEventCount);
   ArrayResize(gRuntimeImportances, gRuntimeEventCount);
   ArrayResize(gRuntimeFeatures, gRuntimeEventCount * gFeatureCount);
   ArrayResize(gDetailEventTimes, gRuntimeEventCount);
   ArrayResize(gDetailEventNames, gRuntimeEventCount);
   ArrayResize(gDetailCountryCodes, gRuntimeEventCount);
   ArrayResize(gDetailImportances, gRuntimeEventCount);

   int detail_handle = OpenCommonCsv(gRuntimeDetailFileName, FILE_READ | FILE_CSV | FILE_ANSI);
   if(detail_handle != INVALID_HANDLE)
     {
      if(!FileIsEnding(detail_handle))
        {
         while(!FileIsLineEnding(detail_handle) && !FileIsEnding(detail_handle))
            FileReadString(detail_handle);

         int idx = 0;
         while(!FileIsEnding(detail_handle) && idx < gRuntimeEventCount)
           {
            string bar_time_str = FileReadString(detail_handle);
            if(StringLen(bar_time_str) == 0 && FileIsEnding(detail_handle))
               break;

            string event_time_str = FileReadString(detail_handle);
            string event_name = FileReadString(detail_handle);
            string country_code = FileReadString(detail_handle);
            string importance_str = FileReadString(detail_handle);

            int parsed_importance = (int)StringToInteger(importance_str);
            if((parsed_importance < 1 || parsed_importance > 3) && StringLen(country_code) > 0)
              {
               event_name = event_name + "," + country_code;
               country_code = importance_str;
               importance_str = FileReadString(detail_handle);
               parsed_importance = (int)StringToInteger(importance_str);
              }

            gDetailEventTimes[idx] = StringToTime(event_time_str);
            gDetailEventNames[idx] = SanitizeCsvText(event_name);
            gDetailCountryCodes[idx] = SanitizeCsvText(country_code);
            gDetailImportances[idx] = parsed_importance;

            while(!FileIsLineEnding(detail_handle) && !FileIsEnding(detail_handle))
               FileReadString(detail_handle);
            idx++;
           }
        }
      FileClose(detail_handle);
     }
   else
     {
      Print("Failed to open runtime event detail file: ", gRuntimeDetailFileName, " Err: ", GetLastError());
     }

   if(gRuntimeEventCount <= 0)
     {
      Print("No runtime events loaded from ", gRuntimeFileName);
      return false;
     }

   Print("Loaded ", gRuntimeEventCount, " runtime event bars from ", gRuntimeFileName);
   if(gRuntimeEventCount > 0)
     {
      Print("Runtime event range | first=",
            TimeToString(gRuntimeBarTimes[0], TIME_DATE | TIME_MINUTES),
            " | last=",
            TimeToString(gRuntimeBarTimes[gRuntimeEventCount - 1], TIME_DATE | TIME_MINUTES),
            " | features=",
            gFeatureCount);
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
bool EvaluateEvent(const int event_index, double &prediction)
  {
   vectorf features(gFeatureCount);
   for(int i = 0; i < gFeatureCount; i++)
      features[i] = (float)GetRuntimeFeature(event_index, i);

   if(IsEnsembleModelType())
     {
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
      return true;
     }

   return RunPredictionForHandle(onnx_handle, ResolveModelString(), features, prediction);
  }

//+------------------------------------------------------------------+
string ResolveEventComment(const int event_index)
  {
   string event_name = gDetailEventNames[event_index];
   if(StringLen(event_name) == 0)
      event_name = "Economic Event";
   return SanitizeCsvText(event_name);
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
double CalculateOrderLotSize()
  {
   double min_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_volume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double volume_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

   if(min_volume <= 0.0)
      min_volume = 0.01;
   if(max_volume < min_volume)
      max_volume = min_volume;
   if(volume_step <= 0.0)
      volume_step = min_volume;

   if(InpLotMode == LOT_MODE_STATIC)
     {
      double static_lot = InpLotSize;
      if(static_lot < min_volume)
         static_lot = min_volume;
      if(static_lot > max_volume)
         static_lot = max_volume;
      int static_digits = 2;
      if(volume_step > 0.0)
         static_digits = (int)MathMax(0.0, MathRound(-MathLog10(volume_step)));
      return NormalizeDouble(static_lot, static_digits);
     }

   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);

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
   while(gNextEventIndex < gRuntimeEventCount && gRuntimeBarTimes[gNextEventIndex] < previous_bar_time)
      gNextEventIndex++;

   int min_importance = ResolveMinimumImportance();
   int best_signal = 0;
   double best_prediction = 0.0;
   int best_event_index = -1;
   bool event_found_in_window = false;

   int idx = gNextEventIndex;
   while(idx < gRuntimeEventCount && gRuntimeBarTimes[idx] < current_bar_time)
     {
      if(gRuntimeBarTimes[idx] >= previous_bar_time)
        {
         event_found_in_window = true;
         if(InpTraceEventWindows)
            DebugPrint(StringFormat(
               "Runtime event matched window | previous=%s | current=%s | event_bar=%s | event=%s | importance=%d | min_importance=%d",
               TimeToString(previous_bar_time, TIME_DATE | TIME_MINUTES),
               TimeToString(current_bar_time, TIME_DATE | TIME_MINUTES),
               TimeToString(gRuntimeBarTimes[idx], TIME_DATE | TIME_MINUTES),
               ResolveEventComment(idx),
               gRuntimeImportances[idx],
               min_importance
            ));

         if(gRuntimeImportances[idx] < min_importance)
           {
            if(InpTraceEventWindows)
               DebugPrint(StringFormat(
                  "Runtime event skipped by importance filter | event_bar=%s | importance=%d | min_importance=%d",
                  TimeToString(gRuntimeBarTimes[idx], TIME_DATE | TIME_MINUTES),
                  gRuntimeImportances[idx],
                  min_importance
               ));
            idx++;
            continue;
           }

         double prediction = 0.0;
         if(EvaluateEvent(idx, prediction))
           {
            int signal = 0;
            if(prediction >= gActiveThresholdBuy)
               signal = 1;
            else if(prediction <= gActiveThresholdSell)
               signal = -1;

            DebugPrint(StringFormat(
               "Event evaluated | bar=%s | event=%s | importance=%d | prediction=%.6f | signal=%d",
               TimeToString(gRuntimeBarTimes[idx], TIME_DATE | TIME_MINUTES),
               ResolveEventComment(idx),
               gRuntimeImportances[idx],
               prediction,
               signal
            ));

            if(signal != 0 && (best_event_index < 0 || MathAbs(prediction) > MathAbs(best_prediction)))
              {
               best_signal = signal;
               best_prediction = prediction;
               best_event_index = idx;
              }
           }
         else
           {
            DebugPrint(StringFormat(
               "Event evaluation failed | event_bar=%s | event=%s | err=%d",
               TimeToString(gRuntimeBarTimes[idx], TIME_DATE | TIME_MINUTES),
               ResolveEventComment(idx),
               GetLastError()
            ));
           }
        }
      idx++;
     }

   gNextEventIndex = idx;
   if(best_event_index < 0)
     {
      if(event_found_in_window && InpTraceEventWindows)
         DebugPrint(StringFormat(
            "Runtime event window completed without tradable signal | previous=%s | current=%s | buy_threshold=%.6f | sell_threshold=%.6f",
            TimeToString(previous_bar_time, TIME_DATE | TIME_MINUTES),
            TimeToString(current_bar_time, TIME_DATE | TIME_MINUTES),
            gActiveThresholdBuy,
            gActiveThresholdSell
         ));
      return false;
     }

   if(InpMaxSpread > 0 && current_spread > InpMaxSpread)
     {
      DebugPrint(StringFormat(
         "Spread filter blocked signal | bar=%s | event=%s | spread=%d | max=%d | prediction=%.6f | signal=%d",
         TimeToString(gRuntimeBarTimes[best_event_index], TIME_DATE | TIME_MINUTES),
         ResolveEventComment(best_event_index),
         current_spread,
         InpMaxSpread,
         best_prediction,
         best_signal
      ));
      return false;
     }

   DebugPrint(StringFormat(
      "Best signal selected | bar=%s | event=%s | prediction=%.6f | signal=%d",
      TimeToString(gRuntimeBarTimes[best_event_index], TIME_DATE | TIME_MINUTES),
      ResolveEventComment(best_event_index),
      best_prediction,
      best_signal
   ));

   return ExecuteSignal(best_signal, best_prediction, ResolveEventComment(best_event_index));
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   onnx_handle = INVALID_HANDLE;
   ArrayInitialize(ensemble_handles, INVALID_HANDLE);

   trade.SetExpertMagicNumber(InpMagicNumber);

   gRuntimeFileName = ResolveRuntimeFileName();
   gRuntimeDetailFileName = ResolveRuntimeDetailFileName();
   gTransactionCsvFile = ResolveTransactionCsvFile();
   gThresholdFileName = ResolveThresholdFileName();

   if(!LoadRuntimeEvents())
      return INIT_FAILED;
   if(!LoadModelThresholds())
      return INIT_FAILED;
   if(!InitOnnx())
      return INIT_FAILED;

   Print("ML Regression EA initialized successfully. Symbol: ", _Symbol,
         " | Model: ", ResolveModelString(),
         " | Runtime: ", gRuntimeFileName,
         " | NewsFilterMinImportance: ", ResolveMinimumImportance(),
         " | ThresholdBuy: ", DoubleToString(gActiveThresholdBuy, 6),
         " | ThresholdSell: ", DoubleToString(gActiveThresholdSell, 6),
         " | MaxSpread: ", InpMaxSpread);
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(InpEnableTransactionCsv && InpExportHistoryOnDeinit)
      ExportAllDealHistoryCsv();
   ReleaseModels();
  }

//+------------------------------------------------------------------+
void OnTick()
  {
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
string DealTypeToText(const ENUM_DEAL_TYPE deal_type)
  {
   if(deal_type == DEAL_TYPE_BUY) return "BUY";
   if(deal_type == DEAL_TYPE_SELL) return "SELL";
   if(deal_type == DEAL_TYPE_BALANCE) return "BALANCE";
   if(deal_type == DEAL_TYPE_CREDIT) return "CREDIT";
   if(deal_type == DEAL_TYPE_CHARGE) return "CHARGE";
   if(deal_type == DEAL_TYPE_CORRECTION) return "CORRECTION";
   if(deal_type == DEAL_TYPE_BONUS) return "BONUS";
   if(deal_type == DEAL_TYPE_COMMISSION) return "COMMISSION";
   if(deal_type == DEAL_TYPE_COMMISSION_DAILY) return "COMMISSION_DAILY";
   if(deal_type == DEAL_TYPE_COMMISSION_MONTHLY) return "COMMISSION_MONTHLY";
   if(deal_type == DEAL_TYPE_COMMISSION_AGENT_DAILY) return "COMMISSION_AGENT_DAILY";
   if(deal_type == DEAL_TYPE_COMMISSION_AGENT_MONTHLY) return "COMMISSION_AGENT_MONTHLY";
   if(deal_type == DEAL_TYPE_INTEREST) return "INTEREST";
   if(deal_type == DEAL_TYPE_BUY_CANCELED) return "BUY_CANCELED";
   if(deal_type == DEAL_TYPE_SELL_CANCELED) return "SELL_CANCELED";
   if(deal_type == DEAL_DIVIDEND) return "DIVIDEND";
   if(deal_type == DEAL_DIVIDEND_FRANKED) return "DIVIDEND_FRANKED";
   if(deal_type == DEAL_TAX) return "TAX";
   return "OTHER";
  }

//+------------------------------------------------------------------+
string DealEntryToText(const ENUM_DEAL_ENTRY deal_entry)
  {
   if(deal_entry == DEAL_ENTRY_IN) return "IN";
   if(deal_entry == DEAL_ENTRY_OUT) return "OUT";
   if(deal_entry == DEAL_ENTRY_INOUT) return "INOUT";
   if(deal_entry == DEAL_ENTRY_OUT_BY) return "OUT_BY";
   return "OTHER";
  }

//+------------------------------------------------------------------+
bool AppendDealCsv(const ulong deal_ticket)
  {
   if(!InpEnableTransactionCsv || deal_ticket == 0)
      return false;
   if(!HistoryDealSelect(deal_ticket))
      return false;

   string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
   if(symbol != _Symbol)
      return true;
   if((ulong)HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != InpMagicNumber)
      return true;

   int fh = OpenCommonCsv(gTransactionCsvFile, FILE_READ | FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(fh == INVALID_HANDLE)
     {
      fh = OpenCommonCsv(gTransactionCsvFile, FILE_WRITE | FILE_CSV | FILE_ANSI);
      if(fh == INVALID_HANDLE)
        {
         Print("Cannot open transaction CSV: ", gTransactionCsvFile, " err=", GetLastError());
         return false;
        }
     }

   if(FileSize(fh) == 0)
     {
      FileWrite(fh,
                "time",
                "deal_ticket",
                "order_ticket",
                "position_id",
                "symbol",
                "deal_type",
                "entry",
                "volume",
                "price",
                "sl",
                "tp",
                "profit",
                "swap",
                "commission",
                "comment");
     }

   FileSeek(fh, 0, SEEK_END);
   string comment = SanitizeCsvText(HistoryDealGetString(deal_ticket, DEAL_COMMENT));

   FileWrite(fh,
             TimeToString((datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME), TIME_DATE | TIME_SECONDS),
             (string)deal_ticket,
             (string)HistoryDealGetInteger(deal_ticket, DEAL_ORDER),
             (string)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID),
             symbol,
             DealTypeToText((ENUM_DEAL_TYPE)HistoryDealGetInteger(deal_ticket, DEAL_TYPE)),
             DealEntryToText((ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY)),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_VOLUME), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PRICE), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SL), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_TP), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PROFIT), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SWAP), 2),
             DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION), 2),
             comment);
   FileClose(fh);
   return true;
  }

//+------------------------------------------------------------------+
bool ExportAllDealHistoryCsv()
  {
   if(!InpEnableTransactionCsv)
      return false;
   if(!HistorySelect(0, TimeCurrent()))
      return false;

   int fh = OpenCommonCsv(gTransactionCsvFile, FILE_WRITE | FILE_CSV | FILE_ANSI);
   if(fh == INVALID_HANDLE)
     {
      Print("Cannot rewrite transaction CSV: ", gTransactionCsvFile, " err=", GetLastError());
      return false;
     }

   FileWrite(fh,
             "time",
             "deal_ticket",
             "order_ticket",
             "position_id",
             "symbol",
             "deal_type",
             "entry",
             "volume",
             "price",
             "sl",
             "tp",
             "profit",
             "swap",
             "commission",
             "comment");

   int total = HistoryDealsTotal();
   for(int i = 0; i < total; i++)
     {
      ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket == 0 || !HistoryDealSelect(deal_ticket))
         continue;

      string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
      if(symbol != _Symbol)
         continue;
      if((ulong)HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != InpMagicNumber)
         continue;

      string comment = SanitizeCsvText(HistoryDealGetString(deal_ticket, DEAL_COMMENT));

      FileWrite(fh,
                TimeToString((datetime)HistoryDealGetInteger(deal_ticket, DEAL_TIME), TIME_DATE | TIME_SECONDS),
                (string)deal_ticket,
                (string)HistoryDealGetInteger(deal_ticket, DEAL_ORDER),
                (string)HistoryDealGetInteger(deal_ticket, DEAL_POSITION_ID),
                symbol,
                DealTypeToText((ENUM_DEAL_TYPE)HistoryDealGetInteger(deal_ticket, DEAL_TYPE)),
                DealEntryToText((ENUM_DEAL_ENTRY)HistoryDealGetInteger(deal_ticket, DEAL_ENTRY)),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_VOLUME), 2),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PRICE), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SL), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_TP), (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS)),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_PROFIT), 2),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_SWAP), 2),
                DoubleToString(HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION), 2),
                comment);
     }

   FileClose(fh);
   return true;
  }

//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
  {
   if(!InpEnableTransactionCsv)
      return;
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD || trans.deal == 0)
      return;

   AppendDealCsv(trans.deal);
  }
//+------------------------------------------------------------------+
