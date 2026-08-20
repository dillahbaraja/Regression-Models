#property strict
#property version   "3.00"
#property copyright "OpenAI"

input group    "Download Range"
input datetime  InpFromDate       = D'2018.01.01 00:00'; // Start datetime to download
input datetime  InpToDate         = D'2023.12.31 23:59'; // End datetime to download, 0 = use current server time

input group    "Filters"
input string   InpCountries       = "US;EU;GB;JP;CA;AU;NZ;CH";
input bool     InpSkipHolidays    = true;

input group    "Output"
input string   InpOutputFile      = ""; // Empty = auto-name from date range
input bool     InpUseCommonFolder = true;
input bool     InpOverwrite       = true;
input bool     InpRemoveAfterRun  = true;
input int      InpChunkDays       = 365;

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
bool   g_started = false;
bool   g_done = false;
int    g_rows_written = 0;
int    g_values_seen = 0;
datetime g_effective_to = 0;

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

string DateTag(const datetime value)
{
   if(value <= 0)
      return TimeToString(TimeTradeServer(), TIME_DATE);

   return StringSubstr(TimeToString(value, TIME_DATE), 0, 10);
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
   string from_tag = DateTag(InpFromDate);
   string to_tag = DateTag(g_effective_to);
   return StringFormat("NEWS_%s-%s_%s_.csv", from_tag, to_tag, ImportanceTag());
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

void CalculateEffectiveTo()
{
   g_effective_to = (InpToDate == 0 ? TimeTradeServer() : InpToDate);
   if(g_effective_to < InpFromDate)
      g_effective_to = InpFromDate;
}

string BuildHeader()
{
   return "Id,Datetime,Time,Name,CountryCode,CountryName,Importance,Actual,Forecast,Previous,Impact,Url";
}

bool OpenFile()
{
   int flags = FILE_WRITE | FILE_TXT | FILE_ANSI;
   if(InpOverwrite)
      flags |= FILE_REWRITE;
   if(InpUseCommonFolder)
      flags |= FILE_COMMON;

   string output_file = InpOutputFile;
   if(output_file == "")
      output_file = BuildOutputFileName();
   else if(StringFind(output_file, ".csv") < 0)
      output_file = output_file + ".csv";

   g_file_handle = FileOpen(output_file, flags, ',', CP_UTF8);
   if(g_file_handle == INVALID_HANDLE)
   {
      PrintFormat("Failed to open output file '%s'. Error=%d", output_file, GetLastError());
      return false;
   }

   FileWriteString(g_file_handle, BuildHeader() + "\r\n");
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

datetime ChunkEnd(datetime from_time, datetime end_time)
{
   datetime candidate = from_time + (datetime)(InpChunkDays * 86400);
   if(candidate <= from_time)
      candidate = from_time + 86400;
   if(candidate > end_time)
      candidate = end_time;
   return candidate;
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

bool CollectRange(datetime from_time, datetime to_time)
{
   MqlCalendarValue values[];
   ResetLastError();
   int total = CalendarValueHistory(values, from_time, to_time);
   if(total <= 0)
      return true;

   g_values_seen += total;

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

bool DownloadNews()
{
   MqlCalendarCountry countries[];
   ResetLastError();
   int country_count = CalendarCountries(countries);
   if(country_count <= 0)
   {
      PrintFormat("CalendarCountries() returned %d. Error=%d", country_count, GetLastError());
      return false;
   }

   PrintFormat("CalendarCountries() = %d", country_count);
   Comment("Collecting calendar values by date range...");
   ChartRedraw();

   datetime from_time = InpFromDate;
   while(from_time <= g_effective_to && !IsStopped())
   {
      datetime to_time = ChunkEnd(from_time, g_effective_to);
      if(!CollectRange(from_time, to_time))
         return false;

      if(to_time == g_effective_to)
         break;

      from_time = to_time + 1;
   }

   if(ArraySize(g_rows) > 1)
      SortRows(0, ArraySize(g_rows) - 1);

   return true;
}

bool WriteRows()
{
   for(int i = 0; i < ArraySize(g_rows) && !IsStopped(); i++)
   {
      if(FileWriteString(g_file_handle, g_rows[i].line + "\r\n") == 0)
         return false;

      g_rows_written++;
      if((g_rows_written % 500) == 0)
      {
         string msg = StringFormat("Progress: rows=%d values=%d", g_rows_written, g_values_seen);
         Comment(msg);
         Print(msg);
         ChartRedraw();
      }
   }

   return true;
}

int OnInit()
{
   SplitCountries();
   CalculateEffectiveTo();

   if(ArraySize(g_selected_countries) <= 0)
   {
      Print("InpCountries is empty.");
      return INIT_FAILED;
   }

   EventSetTimer(1);
   Comment("NewsDownloaderEA ready...");
   PrintFormat("Download range: %s -> %s", TimeToString(InpFromDate, TIME_DATE | TIME_SECONDS), TimeToString(g_effective_to, TIME_DATE | TIME_SECONDS));
   return INIT_SUCCEEDED;
}

void OnTimer()
{
   if(g_started || g_done)
      return;

   g_started = true;

   if(!OpenFile())
   {
      Comment("Failed to open output file.");
      return;
   }

   Comment("Downloading news...");
   ChartRedraw();

   bool ok = DownloadNews();
   if(ok)
      ok = WriteRows();

   if(g_file_handle != INVALID_HANDLE)
   {
      FileClose(g_file_handle);
      g_file_handle = INVALID_HANDLE;
   }

   if(ok)
   {
      g_done = true;
      string msg = StringFormat("Done. Rows=%d Values=%d File=%s", g_rows_written, g_values_seen, (InpOutputFile == "" ? BuildOutputFileName() : InpOutputFile));
      Print(msg);
      Comment(msg);
      ChartRedraw();
      if(InpRemoveAfterRun)
         ExpertRemove();
   }
   else
   {
      Comment("Download failed. Check Journal.");
   }
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   if(g_file_handle != INVALID_HANDLE)
   {
      FileClose(g_file_handle);
      g_file_handle = INVALID_HANDLE;
   }
   ArrayFree(g_rows);
   ArrayFree(g_event_cache);
   ArrayFree(g_country_cache);
   Comment("");
}

void OnTick()
{
}
