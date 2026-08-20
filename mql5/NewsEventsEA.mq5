#property strict
#property version   "1.00"
#property copyright "OpenAI"

input string   InpCountries       = "US;EU;GB;JP;CA;AU;NZ;CH";
input bool     InpSkipHolidays    = true;
input bool     InpUseCommonFolder = true;
input bool     InpOverwrite       = true;
input int      InpRefreshSeconds  = 300;

enum NewsImportanceFilter
{
   ALL = 0,
   LOW = 1,
   MODERATE = 2,
   HIGH = 3
};

input NewsImportanceFilter InpMinImportance = ALL;

struct EventCacheItem
{
   ulong id;
   bool  valid;
   MqlCalendarEvent event;
};

struct CountryCacheItem
{
   ulong id;
   bool  valid;
   MqlCalendarCountry country;
};

struct NewsRow
{
   datetime dt;
   ulong    value_id;
   string   line;
};

string g_selected_countries[];
EventCacheItem g_event_cache[];
CountryCacheItem g_country_cache[];
NewsRow g_rows[];
int    g_file_handle = INVALID_HANDLE;
bool   g_refreshing = false;
datetime g_week_start = 0;
datetime g_week_end = 0;
int    g_refresh_interval = 300;

string CsvEscape(const string text)
{
   string out = text;
   StringReplace(out, "\"", "\"\"");
   return "\"" + out + "\"";
}

string DtToText(const datetime value)
{
   if(value <= 0)
      return "";
   return TimeToString(value, TIME_DATE | TIME_SECONDS);
}

string TimeOnly(const datetime value)
{
   if(value <= 0)
      return "";
   return TimeToString(value, TIME_MINUTES);
}

string ValueText(const bool has_value, const double value)
{
   if(!has_value)
      return "";

   string text = DoubleToString(value, 6);
   while(StringLen(text) > 0 && StringGetCharacter(text, StringLen(text) - 1) == '0')
      text = StringSubstr(text, 0, StringLen(text) - 1);
   if(StringLen(text) > 0 && StringGetCharacter(text, StringLen(text) - 1) == '.')
      text = StringSubstr(text, 0, StringLen(text) - 1);
   return text;
}

string ImportanceTag()
{
   if(InpMinImportance == ALL)
      return "ALL";
   if(InpMinImportance == LOW)
      return "LOW";
   if(InpMinImportance == MODERATE)
      return "MEDIUM";
   if(InpMinImportance == HIGH)
      return "HIGH";
   return "ALL";
}

string BuildOutputFileName()
{
   return "NewsEvents.csv";
}

bool IsImportanceAllowed(const ENUM_CALENDAR_EVENT_IMPORTANCE importance)
{
   if(InpMinImportance == ALL)
      return true;
   if(InpMinImportance == LOW)
      return importance >= CALENDAR_IMPORTANCE_LOW;
   if(InpMinImportance == MODERATE)
      return importance >= CALENDAR_IMPORTANCE_MODERATE;
   if(InpMinImportance == HIGH)
      return importance >= CALENDAR_IMPORTANCE_HIGH;
   return true;
}

bool IsSelectedCountry(const string code)
{
   for(int i = 0; i < ArraySize(g_selected_countries); i++)
   {
      if(g_selected_countries[i] == code)
         return true;
   }
   return false;
}

void SplitCountries()
{
   int n = StringSplit(InpCountries, ';', g_selected_countries);
   for(int i = 0; i < n; i++)
   {
      string code = g_selected_countries[i];
      StringTrimLeft(code);
      StringTrimRight(code);
      g_selected_countries[i] = code;
   }
}

void GetCurrentWeekRange(datetime &week_start, datetime &week_end)
{
   MqlDateTime dt;
   TimeToStruct(TimeTradeServer(), dt);

   int day_of_week = dt.day_of_week; // 0 = Sunday, 1 = Monday, ..., 6 = Saturday
   int days_from_monday = (day_of_week == 0 ? 6 : day_of_week - 1);

   dt.hour = 0;
   dt.min = 0;
   dt.sec = 0;

   week_start = StructToTime(dt) - (days_from_monday * 86400);
   week_end   = week_start + (4 * 86400) + 86399; // Friday 23:59:59
}

bool IsWeekday()
{
   MqlDateTime dt;
   TimeToStruct(TimeTradeServer(), dt);
   return (dt.day_of_week >= 1 && dt.day_of_week <= 5);
}

string BuildHeader()
{
   return "Id,Datetime,Time,Name,CountryCode,CountryName,Importance,Actual,Forecast,Previous,Impact,Url";
}

bool OpenFile(int &handle, string &output_file)
{
   int flags = FILE_WRITE | FILE_TXT | FILE_ANSI;
   if(InpOverwrite)
      flags |= FILE_REWRITE;
   if(InpUseCommonFolder)
      flags |= FILE_COMMON;

   output_file = BuildOutputFileName();
   handle = FileOpen(output_file, flags, ',', CP_UTF8);
   if(handle == INVALID_HANDLE)
   {
      PrintFormat("Failed to open output file '%s'. Error=%d", output_file, GetLastError());
      return false;
   }

   FileWriteString(handle, BuildHeader() + "\r\n");
   PrintFormat("Output file: %s", output_file);
   return true;
}

bool GetCachedEvent(const ulong event_id, MqlCalendarEvent &event)
{
   for(int i = 0; i < ArraySize(g_event_cache); i++)
   {
      if(g_event_cache[i].valid && g_event_cache[i].id == event_id)
      {
         event = g_event_cache[i].event;
         return true;
      }
   }

   if(!CalendarEventById(event_id, event))
      return false;

   int size = ArraySize(g_event_cache);
   ArrayResize(g_event_cache, size + 1);
   g_event_cache[size].id = event_id;
   g_event_cache[size].valid = true;
   g_event_cache[size].event = event;
   return true;
}

bool GetCachedCountry(const ulong country_id, MqlCalendarCountry &country)
{
   for(int i = 0; i < ArraySize(g_country_cache); i++)
   {
      if(g_country_cache[i].valid && g_country_cache[i].id == country_id)
      {
         country = g_country_cache[i].country;
         return true;
      }
   }

   if(!CalendarCountryById(country_id, country))
      return false;

   int size = ArraySize(g_country_cache);
   ArrayResize(g_country_cache, size + 1);
   g_country_cache[size].id = country_id;
   g_country_cache[size].valid = true;
   g_country_cache[size].country = country;
   return true;
}

string BuildRow(const MqlCalendarCountry &country,
                const MqlCalendarEvent &event,
                const MqlCalendarValue &value)
{
   string row = "";
   row += (string)value.id + ",";
   row += CsvEscape(DtToText(value.time)) + ",";
   row += CsvEscape(TimeOnly(value.time)) + ",";
   row += CsvEscape(event.name) + ",";
   row += CsvEscape(country.code) + ",";
   row += CsvEscape(country.name) + ",";
   row += (string)event.importance + ",";
   row += ValueText(value.HasActualValue(), value.GetActualValue()) + ",";
   row += ValueText(value.HasForecastValue(), value.GetForecastValue()) + ",";
   row += ValueText(value.HasPreviousValue(), value.GetPreviousValue()) + ",";
   row += (string)((int)value.impact_type) + ",";
   row += CsvEscape(event.source_url);
   return row;
}

void AppendRow(const MqlCalendarValue &value,
               const MqlCalendarEvent &event,
               const MqlCalendarCountry &country)
{
   int size = ArraySize(g_rows);
   ArrayResize(g_rows, size + 1);
   g_rows[size].dt = value.time;
   g_rows[size].value_id = value.id;
   g_rows[size].line = BuildRow(country, event, value);
}

void SwapRows(int i, int j)
{
   NewsRow tmp = g_rows[i];
   g_rows[i] = g_rows[j];
   g_rows[j] = tmp;
}

bool RowLess(const NewsRow &a, const NewsRow &b)
{
   if(a.dt < b.dt) return true;
   if(a.dt > b.dt) return false;
   return a.value_id < b.value_id;
}

void SortRows(int left, int right)
{
   int i = left;
   int j = right;
   NewsRow pivot = g_rows[(left + right) / 2];

   while(i <= j)
   {
      while(RowLess(g_rows[i], pivot)) i++;
      while(RowLess(pivot, g_rows[j])) j--;
      if(i <= j)
      {
         SwapRows(i, j);
         i++;
         j--;
      }
   }

   if(left < j)  SortRows(left, j);
   if(i < right) SortRows(i, right);
}

bool CollectWeekRange()
{
   MqlCalendarValue values[];
   ResetLastError();
   int total = CalendarValueHistory(values, g_week_start, g_week_end);
   if(total <= 0)
   {
      PrintFormat("CalendarValueHistory() returned %d. Error=%d", total, GetLastError());
      return false;
   }

   for(int i = 0; i < total && !IsStopped(); i++)
   {
      MqlCalendarEvent event;
      if(!GetCachedEvent(values[i].event_id, event))
         continue;

      if(InpSkipHolidays && event.type == CALENDAR_TYPE_HOLIDAY)
         continue;

      if(!IsImportanceAllowed((ENUM_CALENDAR_EVENT_IMPORTANCE)event.importance))
         continue;

      MqlCalendarCountry country;
      if(!GetCachedCountry(event.country_id, country))
         continue;

      if(!IsSelectedCountry(country.code))
         continue;

      AppendRow(values[i], event, country);
   }

   return true;
}

bool RebuildFile()
{
   ArrayFree(g_rows);

   GetCurrentWeekRange(g_week_start, g_week_end);

   int handle = INVALID_HANDLE;
   string output_file = "";
   if(!OpenFile(handle, output_file))
      return false;

   bool ok = CollectWeekRange();
   if(ok && ArraySize(g_rows) > 1)
      SortRows(0, ArraySize(g_rows) - 1);

   if(ok)
   {
      for(int i = 0; i < ArraySize(g_rows) && !IsStopped(); i++)
      {
         if(FileWriteString(handle, g_rows[i].line + "\r\n") == 0)
         {
            ok = false;
            break;
         }

         if((i + 1) % 500 == 0)
         {
            string msg = StringFormat("Progress: rows=%d", i + 1);
            Comment(msg);
            Print(msg);
            ChartRedraw();
         }
      }
   }

   FileClose(handle);

   if(ok)
   {
      string msg = StringFormat("Done. Rows=%d Week=%s to %s File=%s",
                                ArraySize(g_rows),
                                DtToText(g_week_start),
                                DtToText(g_week_end),
                                output_file);
      Print(msg);
      Comment(msg);
      ChartRedraw();
   }

   return ok;
}

void RunRefresh()
{
   if(g_refreshing)
      return;

   if(!IsWeekday())
      return;

   g_refreshing = true;
   RebuildFile();
   g_refreshing = false;
}

int OnInit()
{
   SplitCountries();
   if(ArraySize(g_selected_countries) <= 0)
   {
      Print("InpCountries is empty.");
      return INIT_FAILED;
   }

   g_refresh_interval = (InpRefreshSeconds < 1 ? 300 : InpRefreshSeconds);

   EventSetTimer(g_refresh_interval);
   Comment("NewsEventsEA ready...");

   RunRefresh();
   return INIT_SUCCEEDED;
}

void OnTimer()
{
   RunRefresh();
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   ArrayFree(g_rows);
   ArrayFree(g_event_cache);
   ArrayFree(g_country_cache);
   Comment("");
}

void OnTick()
{
}
