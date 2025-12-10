# Performance Optimization Plan - OI_Gemini Application

## Executive Summary

**Current Performance Issues:**
- Login page load time: **>60 seconds**
- Option chain dashboard load time: **~40 seconds**
- Total time to usable interface: **~100 seconds**

**Target Performance Goals:**
- Login page load time: **<2 seconds**
- Option chain dashboard load time: **<5 seconds**
- Total time to usable interface: **<10 seconds**

---

## Phase 1: Diagnosis & Profiling

### 1.1 Add Performance Instrumentation

**Objective:** Identify exact bottlenecks with timing data.

**Actions:**
1. **Add timing decorators** to key functions:
   - `initialize_system()` - Full initialization
   - `_bootstrap_initial_prices()` - Price bootstrap
   - `reel_persistence.load()` - Historical data loading
   - `_ensure_historical_bootstrap()` - Historical bootstrap per exchange
   - `_fetch_all_instruments()` - Instrument fetching
   - `configure_exchange_handlers()` - Handler configuration
   - ML model loading functions
   - Database query functions

2. **Add request timing middleware** for Flask routes:
   - Log time for `/login` GET request
   - Log time for `/` (index) GET request
   - Log time for `/api/data/<exchange>` requests
   - Log time for WebSocket connection establishment

3. **Add database query timing**:
   - Time all `reel_persistence.load()` queries
   - Time historical bootstrap queries
   - Time position loading queries
   - Log slow queries (>1 second)

4. **Create performance dashboard**:
   - Add `/api/performance-metrics` endpoint
   - Track initialization times
   - Track API response times
   - Track database query times

**Expected Output:**
- Detailed timing breakdown of initialization process
- Identification of top 5 slowest operations
- Database query performance metrics

---

## Phase 2: Login Page Optimization

### 2.1 Current Issues Analysis

**Potential Bottlenecks:**
1. **System initialization blocking login page render**
   - Login page may be waiting for system state checks
   - Database connections may be initialized on import
   - Heavy imports may slow down Flask app startup

2. **Template rendering delays**
   - External resource loading (fonts, CSS)
   - Heavy template processing

3. **Session/authentication checks**
   - Database queries for session validation
   - Heavy authentication logic

### 2.2 Optimization Strategies

#### 2.2.1 Lazy System Initialization
- **Current:** System may initialize on module import
- **Fix:** Ensure `initialize_system()` only runs when credentials provided
- **Action:**
  - Verify no heavy operations in module-level code
  - Move database initialization to lazy loading
  - Defer ML model loading until after login

#### 2.2.2 Optimize Login Route
- **Current:** `/login` GET may perform heavy checks
- **Fix:** Minimize work in GET handler
- **Action:**
  ```python
  @app.route('/login', methods=['GET'])
  def login():
      # Minimal checks only
      if session.get('authenticated') and system_state.get('initialized'):
          return redirect(url_for('index'))
      return render_template('login.html', ...)  # Fast render
  ```

#### 2.2.3 Template Optimization
- **Current:** External resources may block rendering
- **Fix:** Optimize resource loading
- **Action:**
  - Preload critical CSS inline
  - Use font-display: swap for fonts
  - Minimize external dependencies
  - Add resource hints (preconnect, dns-prefetch)

#### 2.2.4 Database Connection Pooling
- **Current:** Database connections may be created on import
- **Fix:** Lazy connection pool initialization
- **Action:**
  - Initialize connection pool only when needed
  - Use connection pooling effectively
  - Avoid connection creation in module-level code

**Expected Improvement:** Login page load time: **60s → <2s**

---

## Phase 3: Dashboard Load Optimization

### 3.1 Current Issues Analysis

**Potential Bottlenecks:**
1. **Historical data loading** (`reel_persistence.load()`)
   - Loading 40 minutes of historical data per exchange
   - Multiple exchanges = multiple large queries
   - No pagination or limit on data loaded

2. **Historical bootstrap** (`_ensure_historical_bootstrap()`)
   - Fetching historical OI data from Zerodha API
   - Sequential API calls per exchange
   - No caching of bootstrap data

3. **ML model loading**
   - Loading LightGBM models on initialization
   - Model files may be large
   - No lazy loading

4. **Database queries on dashboard load**
   - Multiple queries for initial data
   - No query optimization
   - No result caching

5. **WebSocket connection establishment**
   - WebSocket may be connecting during dashboard load
   - Subscription to many tokens
   - No connection pooling

### 3.2 Optimization Strategies

#### 3.2.1 Optimize Historical Data Loading

**Current Problem:**
- `reel_persistence.load()` loads 40 minutes of data per exchange
- For 4 exchanges (NSE, BSE, NSE_MONTHLY, BANKNIFTY_MONTHLY), this could be 4 large queries
- Each query may scan thousands of rows

**Solutions:**

1. **Limit Historical Data Load**
   - Load only last 5-10 minutes for initial display
   - Load remaining data asynchronously in background
   - Use pagination for historical data

2. **Optimize Database Queries**
   - Add proper indexes on `(exchange, timestamp)` for `option_chain_snapshots`
   - Use `LIMIT` clause to restrict initial load
   - Use materialized views for aggregated data

3. **Implement Incremental Loading**
   - Load only data since last session
   - Use timestamp-based filtering
   - Cache last loaded timestamp

4. **Parallel Loading**
   - Load data for multiple exchanges in parallel
   - Use threading or async for concurrent queries

**Implementation:**
```python
def load_recent_data(handler, minutes=5):
    """Load only recent data for fast initial display."""
    cutoff = now_ist() - timedelta(minutes=minutes)
    # Query with LIMIT and proper index
    # Load remaining data in background thread
```

**Expected Improvement:** Historical data load: **20-30s → 2-3s**

#### 3.2.2 Optimize Historical Bootstrap

**Current Problem:**
- `_ensure_historical_bootstrap()` makes sequential API calls
- Each exchange bootstrap may take 5-10 seconds
- No caching of bootstrap results

**Solutions:**

1. **Cache Bootstrap Results**
   - Cache bootstrap data in memory/Redis
   - Reuse bootstrap data if market hasn't changed significantly
   - Cache for 1-5 minutes

2. **Parallel Bootstrap**
   - Bootstrap multiple exchanges in parallel
   - Use threading for concurrent API calls
   - Don't block dashboard load on bootstrap

3. **Lazy Bootstrap**
   - Don't bootstrap on initialization
   - Bootstrap in background after dashboard loads
   - Show "Loading historical data..." message

4. **Optimize API Calls**
   - Reduce number of API calls
   - Batch requests where possible
   - Use efficient date ranges

**Implementation:**
```python
def bootstrap_async(handler):
    """Bootstrap in background thread."""
    Thread(target=_ensure_historical_bootstrap, args=(handler,), daemon=True).start()
```

**Expected Improvement:** Bootstrap time: **20-30s → 0s (async)** or **5-10s (parallel)**

#### 3.2.3 Optimize ML Model Loading

**Current Problem:**
- Models loaded synchronously during initialization
- Large model files may take time to load
- All models loaded even if not immediately needed

**Solutions:**

1. **Lazy Model Loading**
   - Load models only when first prediction needed
   - Load models in background thread
   - Show "ML models loading..." indicator

2. **Parallel Model Loading**
   - Load models for different exchanges in parallel
   - Use multiprocessing for CPU-bound loading

3. **Model Caching**
   - Cache loaded models in memory
   - Reuse models across requests
   - Preload models during low-traffic periods

4. **Optimize Model Files**
   - Compress model files if possible
   - Use efficient serialization format
   - Consider model quantization

**Implementation:**
```python
def load_models_async(exchange):
    """Load ML models in background."""
    Thread(target=_load_ml_models, args=(exchange,), daemon=True).start()
```

**Expected Improvement:** ML model loading: **5-10s → 0s (async)** or **2-3s (parallel)**

#### 3.2.4 Optimize Database Queries

**Current Problem:**
- Multiple queries on dashboard load
- No query optimization
- No result caching

**Solutions:**

1. **Query Optimization**
   - Add proper indexes (verify existing indexes)
   - Use EXPLAIN ANALYZE to identify slow queries
   - Optimize JOIN operations
   - Use materialized views for complex queries

2. **Result Caching**
   - Cache dashboard data for 5-10 seconds
   - Use Redis or in-memory cache
   - Invalidate cache on data updates

3. **Reduce Query Frequency**
   - Batch multiple queries into one
   - Use single query with multiple exchanges
   - Prefetch data for all exchanges at once

4. **Connection Pooling**
   - Ensure proper connection pool configuration
   - Reuse connections effectively
   - Monitor pool usage

**Implementation:**
```python
@lru_cache(maxsize=1)
def get_dashboard_data_cached():
    """Cache dashboard data for 5 seconds."""
    # Return cached data if available
    # Otherwise fetch and cache
```

**Expected Improvement:** Database queries: **5-10s → 1-2s**

#### 3.2.5 Optimize WebSocket Connection

**Current Problem:**
- WebSocket connection may be establishing during dashboard load
- Subscription to many tokens may be slow
- No connection reuse

**Solutions:**

1. **Lazy WebSocket Connection**
   - Don't block dashboard on WebSocket connection
   - Connect in background
   - Show "Connecting..." indicator

2. **Optimize Token Subscription**
   - Batch token subscriptions
   - Subscribe in chunks
   - Prioritize critical tokens

3. **Connection Pooling**
   - Reuse WebSocket connections
   - Implement connection retry logic
   - Monitor connection health

**Expected Improvement:** WebSocket setup: **5-10s → 0s (async)**

---

## Phase 4: Database Optimization

### 4.1 Index Optimization

**Current State:**
- Verify existing indexes are optimal
- Check for missing indexes on frequently queried columns

**Actions:**
1. **Review Existing Indexes**
   - Check indexes on `option_chain_snapshots`:
     - `(exchange, timestamp)` - for time-based queries
     - `(exchange, strike, option_type, timestamp)` - for specific option queries
   - Check indexes on `ml_features`:
     - `(exchange, timestamp)` - for feature queries

2. **Add Missing Indexes**
   - Add composite indexes for common query patterns
   - Add partial indexes for filtered queries
   - Consider covering indexes for frequent queries

3. **Index Maintenance**
   - Monitor index usage
   - Remove unused indexes
   - Rebuild indexes periodically

**Expected Improvement:** Query time: **30-50% reduction**

### 4.2 Query Optimization

**Actions:**
1. **Analyze Slow Queries**
   - Enable query logging
   - Identify queries taking >1 second
   - Use EXPLAIN ANALYZE for slow queries

2. **Optimize Common Queries**
   - Optimize `reel_persistence.load()` queries
   - Optimize historical bootstrap queries
   - Optimize position loading queries

3. **Use Efficient Query Patterns**
   - Use LIMIT for pagination
   - Use WHERE clauses effectively
   - Avoid SELECT * when possible
   - Use appropriate JOIN types

**Expected Improvement:** Query time: **20-40% reduction**

### 4.3 Data Archiving

**Actions:**
1. **Archive Old Data**
   - Move data older than 30 days to archive tables
   - Keep only recent data in main tables
   - Implement data retention policies

2. **Partition Large Tables**
   - Use TimescaleDB hypertables effectively
   - Partition by time for better performance
   - Drop old partitions automatically

**Expected Improvement:** Query time: **10-20% reduction** (for large datasets)

---

## Phase 5: Caching Strategy

### 5.1 Application-Level Caching

**Actions:**
1. **Cache Dashboard Data**
   - Cache exchange data for 5-10 seconds
   - Invalidate cache on updates
   - Use in-memory cache (dict) or Redis

2. **Cache ML Predictions**
   - Cache predictions for 1 minute
   - Invalidate on new data
   - Reduce redundant model calls

3. **Cache Instrument Data**
   - Cache instrument lists
   - Refresh cache periodically
   - Cache token mappings

4. **Cache Bootstrap Results**
   - Cache historical bootstrap data
   - Reuse bootstrap data within time window
   - Invalidate on market open

**Implementation:**
```python
from functools import lru_cache
from datetime import datetime, timedelta

cache_ttl = {}

def get_cached_data(key, ttl_seconds=5):
    """Get cached data with TTL."""
    if key in cache_ttl:
        data, expiry = cache_ttl[key]
        if datetime.now() < expiry:
            return data
    return None

def set_cached_data(key, data, ttl_seconds=5):
    """Set cached data with TTL."""
    cache_ttl[key] = (data, datetime.now() + timedelta(seconds=ttl_seconds))
```

**Expected Improvement:** Response time: **50-70% reduction** for cached requests

### 5.2 Database Query Caching

**Actions:**
1. **Cache Frequent Queries**
   - Cache dashboard data queries
   - Cache position queries
   - Cache historical data queries

2. **Use Materialized Views**
   - Create materialized views for complex queries
   - Refresh views periodically
   - Use views for dashboard data

**Expected Improvement:** Query time: **30-50% reduction**

---

## Phase 6: Frontend Optimization

### 6.1 Reduce Initial Data Load

**Actions:**
1. **Lazy Load Data**
   - Load only visible exchange data initially
   - Load other exchanges on demand
   - Use pagination for large datasets

2. **Optimize API Calls**
   - Batch API calls
   - Use WebSocket for real-time updates
   - Reduce polling frequency

3. **Client-Side Caching**
   - Cache data in browser localStorage
   - Reuse cached data on page reload
   - Implement cache invalidation

**Expected Improvement:** Initial load: **30-40% reduction**

### 6.2 Optimize Rendering

**Actions:**
1. **Virtual Scrolling**
   - Use virtual scrolling for large option chains
   - Render only visible rows
   - Reduce DOM manipulation

2. **Debounce Updates**
   - Debounce frequent updates
   - Batch DOM updates
   - Use requestAnimationFrame

3. **Optimize CSS/JS**
   - Minify CSS and JavaScript
   - Use CDN for static assets
   - Implement code splitting

**Expected Improvement:** Rendering time: **20-30% reduction**

---

## Phase 7: System Architecture Improvements

### 7.1 Asynchronous Initialization

**Actions:**
1. **Background Initialization**
   - Move heavy operations to background threads
   - Don't block UI on initialization
   - Show progress indicators

2. **Progressive Loading**
   - Load critical data first
   - Load non-critical data later
   - Show "Loading..." for background operations

**Implementation:**
```python
def initialize_system_async():
    """Initialize system in background."""
    Thread(target=initialize_system, daemon=True).start()
    # Return immediately, show progress in UI
```

**Expected Improvement:** Perceived load time: **80-90% reduction**

### 7.2 Connection Pooling

**Actions:**
1. **Database Connection Pool**
   - Ensure proper pool configuration
   - Monitor pool usage
   - Tune pool size

2. **API Connection Pooling**
   - Reuse HTTP connections
   - Implement connection pooling for Zerodha API
   - Cache API responses

**Expected Improvement:** Connection overhead: **50-70% reduction**

---

## Phase 8: Monitoring & Measurement

### 8.1 Performance Monitoring

**Actions:**
1. **Add Performance Metrics**
   - Track initialization times
   - Track API response times
   - Track database query times
   - Track WebSocket latency

2. **Create Performance Dashboard**
   - Display key metrics
   - Alert on performance degradation
   - Track trends over time

3. **Set Performance Targets**
   - Define SLA for each operation
   - Monitor against targets
   - Alert on violations

### 8.2 Continuous Optimization

**Actions:**
1. **Regular Performance Reviews**
   - Weekly performance analysis
   - Identify new bottlenecks
   - Optimize based on data

2. **Load Testing**
   - Simulate user load
   - Identify capacity limits
   - Optimize for peak load

---

## Implementation Priority

### High Priority (Immediate Impact)
1. ✅ **Add performance instrumentation** (Phase 1)
2. ✅ **Optimize login route** (Phase 2.2.2)
3. ✅ **Limit historical data load** (Phase 3.2.1)
4. ✅ **Lazy bootstrap** (Phase 3.2.2)
5. ✅ **Lazy ML model loading** (Phase 3.2.3)
6. ✅ **Add caching** (Phase 5.1)

### Medium Priority (Significant Impact)
7. ✅ **Optimize database queries** (Phase 3.2.4)
8. ✅ **Add database indexes** (Phase 4.1)
9. ✅ **Parallel loading** (Phase 3.2.1, 3.2.2)
10. ✅ **Frontend optimization** (Phase 6)

### Low Priority (Incremental Improvement)
11. ✅ **Data archiving** (Phase 4.3)
12. ✅ **Connection pooling** (Phase 7.2)
13. ✅ **Performance monitoring** (Phase 8)

---

## Expected Overall Improvement

**Before Optimization:**
- Login page: **60+ seconds**
- Dashboard load: **40 seconds**
- Total: **100+ seconds**

**After Optimization:**
- Login page: **<2 seconds** (97% improvement)
- Dashboard load: **<5 seconds** (87% improvement)
- Total: **<10 seconds** (90% improvement)

---

## Risk Mitigation

### 1. Core Functionality Preservation
- All optimizations maintain existing functionality
- No changes to business logic
- Backward compatible changes only

### 2. Testing Strategy
- Test each optimization independently
- Verify functionality after each change
- Performance testing before/after

### 3. Rollback Plan
- Keep original code in version control
- Feature flags for new optimizations
- Gradual rollout

---

## Success Criteria

1. ✅ Login page loads in **<2 seconds**
2. ✅ Dashboard loads in **<5 seconds**
3. ✅ All existing functionality works
4. ✅ No degradation in data accuracy
5. ✅ Performance improvements measurable

---

## Next Steps

1. **Review and approve this plan**
2. **Start with Phase 1 (Diagnosis)** to identify exact bottlenecks
3. **Implement high-priority optimizations first**
4. **Measure improvements after each phase**
5. **Iterate based on results**

---

## Notes

- This plan focuses on **non-invasive optimizations** that don't change core functionality
- All optimizations are **backward compatible**
- Performance improvements are **additive** - each optimization builds on previous ones
- **Measurement is critical** - verify improvements with actual timing data

