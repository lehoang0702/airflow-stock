const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(require('node:path').join(__dirname, '../dashboard/static/app.js'), 'utf8');
const elements = new Map();
const context = vm.createContext({document: {getElementById(id) {
  if (!elements.has(id)) elements.set(id, {textContent: '', style: {}});
  return elements.get(id);
}}});
vm.runInContext(source.slice(source.indexOf('function updateKeyStatistics('), source.indexOf('async function loadTickerChart(')), context);
context.res = {candles: [
  {date:'2026-10-05', open:100, close:105, high:110, low:90, volume:1000},
  {date:'2026-10-06', open:106, close:107, high:111, low:100, volume:2000}],
  statistics: {status:'ok', market_cap:1882301399040, currency:'USD', beta:1.185,
    pe_ttm:27.84024, eps_ttm:26.54, bid:718, ask:742.89, low_52w:520.26, high_52w:779.82,
    average_volume:19329054, dividend_rate:2.1, dividend_yield_pct:.28,
    earnings_date:'2026-10-28', earnings_date_end:'2026-10-28', earnings_is_estimate:true,
    ex_dividend_date:'2026-09-21', target_mean_price:794.9607}};
vm.runInContext("updateKeyStatistics('META', res)", context);
const value = id => elements.get(id).textContent;
assert.equal(value('stat-bid'), '718.00');
assert.equal(value('stat-ask'), '742.89');
assert.equal(value('stat-market-cap'), '1.88 nghìn tỷ USD');
assert.equal(value('stat-pe'), '27.84');
assert.equal(value('stat-eps'), '26.54');
assert.equal(value('stat-beta'), '1.19');
assert.equal(value('stat-52w-range'), '520.26 - 779.82'); // With only two candles, use Yahoo's full year.
assert.equal(value('stat-dividend'), '2.10 (0.28%)'); // No extra factor of 100.
assert.equal(value('stat-earnings'), '28/10/2026');
assert.equal(value('stat-target-est'), '794.96');
assert.equal(value('stat-prev-close'), '105.00');
assert.match(value('stat-source-note'), /Yahoo Finance/);
assert.match(value('stat-avg-volume-label'), /3 tháng/);
context.res.statistics = {status:'unavailable', message:'Yahoo temporarily unavailable', eps_ttm:0, dividend_rate:0, dividend_yield_pct:0};
vm.runInContext("updateKeyStatistics('OTHER', res)", context);
assert.equal(value('stat-market-cap'), '--'); // No leftover statistics from META.
assert.equal(value('stat-pe'), '--');
assert.equal(value('stat-eps'), '0.00');
assert.equal(value('stat-dividend'), '0.00 (0.00%)');
assert.equal(value('stat-source-note'), 'Yahoo temporarily unavailable');
assert.equal(value('stat-52w-range'), '--'); // Never invent a full-year range from two days.
assert.match(value('stat-avg-volume-label'), /60 phiên/);
assert.equal(value('stat-open'), '106.00'); // Price data survives metadata failure.
console.log('Quote statistics: real fields, yield units, dates, missing values and ticker switching passed.');
