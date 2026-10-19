// Exercise the real renderer with hostile provider data and invalid links.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.style = {}; }
  append(...children) { this.children.push(...children); }
  replaceChildren() { this.children = []; }
  set innerHTML(_) { throw new Error('External news must never be parsed as HTML'); }
}
const container = new Element('div');
const source = fs.readFileSync(require('node:path').join(__dirname, '../dashboard/static/app.js'), 'utf8');
const start = source.indexOf('function renderNews(');
const renderer = source.slice(start, source.indexOf('// Vẽ biểu đồ nến SVG', start));
const context = vm.createContext({
  document: { getElementById: () => container, createElement: tag => new Element(tag) },
  state: { activeTicker: 'AAPL' }, URL,
});
vm.runInContext(renderer, context);
context.news = [
  { headline: '<img src=x onerror=alert(1)>', source: '<script>bad()</script>', url: 'javascript:alert(1)', date: '2026-10-07' },
  { headline: 'Valid article', url: 'https://example.com/article?x=1&y=2' },
  { headline: 'Broken URL', url: 'https://[' }, null,
];
vm.runInContext('renderNews(news)', context);
assert.equal(container.children.length, 3);
const [hostile, valid, invalid] = container.children;
assert.equal(hostile.children[0].textContent, '<img src=x onerror=alert(1)>');
assert.equal(hostile.children[1].children[0].textContent, '📰 <script>bad()</script>');
assert.equal(hostile.href, 'https://finance.yahoo.com/quote/AAPL/news/');
assert.equal(valid.href, 'https://example.com/article?x=1&y=2');
assert.equal(invalid.href, hostile.href);
assert.equal(hostile.rel, 'noopener noreferrer');
vm.runInContext('renderNews([])', context);
assert.equal(container.children.length, 1);
assert.equal(container.children[0].className, 'empty-news');
vm.runInContext("renderNews([], {status: 'source_error'})", context);
assert.match(container.children[0].textContent, /lỗi kết nối/);

const detail = new Element('div');
const gauge = new Element('span');
const signalBar = new Element('div');
const statusContext = vm.createContext({
  document: { getElementById: id => ({'side-finbert-status': detail, 'side-bert-val': gauge, 'side-bert-bar': signalBar})[id] || null,
              createElement: tag => new Element(tag) },
});
vm.runInContext(source.slice(source.indexOf('function finbertSentimentScores('), start), statusContext);
statusContext.pred = {finbert_message: '<img src=x onerror=alert(1)>',
  finbert_status: 'available', finbert_sources: ['Yahoo Finance RSS'],
  finbert_sentiment: {sentiment_positive:.7, sentiment_negative:.1, sentiment_neutral:.2}};
vm.runInContext('renderFinbertStatus(pred)', statusContext);
assert.equal(detail.hidden, false);
assert.equal(gauge.textContent, '80.0%');
assert.equal(signalBar.style.width, '80.0%');
assert.equal(detail.children[0].textContent, 'FinBERT: <img src=x onerror=alert(1)>');
assert.match(detail.children[1].textContent, /Cảm xúc tin tức: Tích cực 70.0%/);
assert.match(detail.children[2].textContent, /Tăng 80.0% · Giảm 20.0%/);
assert.match(detail.children[3].textContent, /Yahoo Finance RSS/);
vm.runInContext('renderFinbertStatus(null)', statusContext);
assert.equal(detail.hidden, true);
assert.equal(gauge.textContent, 'Không khả dụng');
assert.equal(signalBar.style.width, '0%');
statusContext.pred = {finbert_sentiment: {sentiment_positive: .8, sentiment_negative: .8, sentiment_neutral: .8}};
vm.runInContext('renderFinbertStatus(pred)', statusContext);
assert.equal(gauge.textContent, 'Không khả dụng');
assert.equal(signalBar.style.width, '0%');
console.log('News and FinBERT sentiment renderers passed.');
