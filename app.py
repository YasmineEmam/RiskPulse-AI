import streamlit as st
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import hashlib
import json
import re
import urllib.parse
import feedparser
import numpy as np
import requests
from bs4 import BeautifulSoup

from rapidfuzz import fuzz

# =========================================================
# Web Text Extraction Utilities
# =========================================================
try:
    import trafilatura
    HAS_TRAFILATURA = True
except Exception:
    HAS_TRAFILATURA = False

try:
    from newspaper import Article as NewspaperArticle
    HAS_NEWSPAPER = True
except Exception:
    HAS_NEWSPAPER = False

try:
    from camel_tools.utils.normalize import (
        normalize_unicode,
        normalize_alef_ar,
        normalize_alef_maksura_ar,
        normalize_teh_marbuta_ar,
    )
    from camel_tools.utils.dediac import dediac_ar
    HAS_CAMEL = True
except Exception:
    HAS_CAMEL = False

# =========================================================
# System Constants
# =========================================================
APP_NAME = "RiskPulse AI"
DEFAULT_LOOKBACK_YEARS = 5

# =========================================================
# Data Models
# =========================================================
@dataclass
class Article:
    title: str
    full_text: str
    url: str
    source: str
    date: str
    content_quality: float = 1.0

@dataclass
class RiskEvent:
    domain_ar: str
    matched_keywords: List[str]
    evidence_sentence: str
    severity: int
    score_contribution: float
    relevance: float = 0.0
    source_reliability: float = 0.6
    entity_confidence: float = 0.5
    recency_factor: float = 0.7

@dataclass
class RiskAnalysisResult:
    has_risk: bool
    article: Article
    score: float
    reason: str
    events: List[RiskEvent]
    confidence: float

# =========================================================
# Company Registry
# =========================================================
class CompanyRegistry:
    REGISTRY = {
        "المصرية للاتصالات (WE)": ["المصرية للاتصالات", "we", "وي", "telecom egypt"],
        "شركة القناة لتوزيع الكهرباء": ["القناة", "كهرباء القناة"],
        "ماريدايف": ["ماريدايف", "maridive"],
        "الكابلات الكهربائية المصرية": ["الكابلات الكهربائية"],
        "مدينة مصر للإسكان": ["مدينة مصر", "madinet masr"],
        "Premium Healthcare": ["premium healthcare", "بريميوم هيلثكير"],
        "مينا فارم": ["مينا فارم", "minapharm"],
        "جنوب الوادي للأسمنت": ["جنوب الوادي"],
        "السويدي إليكتريك": ["مجموعة السويدي", "elsewedy", "السويدى اليكتريك"],
        "جهينة": ["جهينة", "juhayna"],
        "إيديتا": ["إيديتا", "edita"],
        "طلعت مصطفى": ["طلعت مصطفى", "tmg"],
        "أوراسكوم": ["اوراسكوم", "أوراسكوم للتنمية", "orascom", "أوراسكوم كونستراكشون", "أوراسكوم للاستثمار", "أوراسكوم المالية"]
    }

# =========================================================
# Arabic Normalization Layer
# =========================================================
class ArabicProcessor:
    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        if HAS_CAMEL:
            try:
                text = normalize_unicode(text)
                text = dediac_ar(text)
                text = normalize_alef_ar(text)
                text = normalize_alef_maksura_ar(text)
                text = normalize_teh_marbuta_ar(text)
            except Exception:
                pass
        else:
            text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
            text = re.sub(r"[إأآٱ]", "ا", text)
            text = text.replace("ى", "ي").replace("ة", "ه")
        
        text = text.replace("ـ", " ")
        text = re.sub(r"[^\u0600-\u06FF\w\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

# =========================================================
# Corporate Entity Resolution
# =========================================================
def resolve_company(user_input: str) -> Tuple[str, List[str]]:
    normalized_input = ArabicProcessor.normalize(user_input)
    if not normalized_input:
        return user_input.strip(), [user_input.strip()]

    # 1. مطابقة تامة أولاً للأسماء والبدائل
    for official, aliases in CompanyRegistry.REGISTRY.items():
        names = [official] + aliases
        for name in names:
            normalized_name = ArabicProcessor.normalize(name)
            if normalized_name == normalized_input:
                return official, names

    # 2. مطابقة تقريبية (Fuzzy Matching) لأفضل نتيجة مطابقة
    best_official = user_input.strip()
    best_aliases = [user_input.strip()]
    best_score = 0

    for official, aliases in CompanyRegistry.REGISTRY.items():
        names = [official] + aliases
        for name in names:
            normalized_name = ArabicProcessor.normalize(name)
            if not normalized_name:
                continue
            score = fuzz.token_set_ratio(normalized_input, normalized_name)
            if score > best_score:
                best_score = score
                best_official = official
                best_aliases = names

    if best_score >= 80:
        return best_official, best_aliases

    return user_input.strip(), [user_input.strip()]

def calculate_entity_confidence(sentence: str, company_name: str, aliases: List[str]) -> float:
    normalized_sentence = ArabicProcessor.normalize(sentence)
    if not normalized_sentence:
        return 0.0
    names = [company_name] + aliases
    best = 0
    for alias in names:
        normalized_alias = ArabicProcessor.normalize(alias)
        if not normalized_alias:
            continue
        if normalized_alias in normalized_sentence:
            return 1.0
        best = max(best, fuzz.partial_ratio(normalized_alias, normalized_sentence))
    if best >= 80: return 0.9
    if best >= 65: return 0.8
    return 0.6

# =========================================================
# Multi-Source News Aggregator
# =========================================================
class AdvancedNewsAggregator:
    @staticmethod
    def _fetch_rss_safely(url: str) -> dict:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=8)
            if response.status_code == 200:
                return feedparser.parse(response.content)
        except Exception:
            pass
        return {}

    @staticmethod
    def extract_text_fallback(url: str, raw_summary: str, timeout: int = 6) -> Tuple[str, float]:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        
        if HAS_TRAFILATURA:
            try:
                resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                if resp.status_code == 200:
                    text = trafilatura.extract(resp.text, include_comments=False)
                    if text and len(text.split()) >= 50:
                        return text.strip(), min(1.0, len(text.split()) / 400.0)
            except Exception:
                pass

        try:
            resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, "html.parser")
                paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text().strip()) > 20]
                text = " ".join(paragraphs)
                if len(text.split()) >= 50:
                    return text.strip(), min(0.9, len(text.split()) / 400.0)
        except Exception:
            pass

        if raw_summary:
            clean_summary = BeautifulSoup(raw_summary, "html.parser").get_text(" ").strip()
            if len(clean_summary) > 25:
                return clean_summary, 0.75

        return "", 0.0

    @staticmethod
    def fetch_news(company_name: str, aliases: List[str], max_results: int = 20, max_workers: int = 8) -> List[Article]:
        candidates = {}
        queries = [
            f'"{company_name}" أزمة OR خسائر OR قضايا OR غرامة OR تعثر OR ديون OR نزاع OR تحكيم when:5y',
            f'"{company_name}" مصر when:2y',
            f'"{company_name}" when:1y'
        ]
        
        if aliases:
            for alt in aliases[:2]:
                if alt != company_name:
                    queries.append(f'"{alt}" أزمة OR خسائر OR قضايا OR نزاع when:5y')
                    queries.append(f'"{alt}" when:2y')

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures_ordered = []
            for q in queries:
                encoded = urllib.parse.quote(q)
                rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=ar&gl=EG&ceid=EG:ar"
                futures_ordered.append(executor.submit(AdvancedNewsAggregator._fetch_rss_safely, rss_url))

            # Process in exact order of priority (Crisis query first, then generic)
            for future in futures_ordered:
                try:
                    feed = future.result()
                    for entry in feed.get("entries", []):
                        link = entry.get("link", "")
                        title = entry.get("title", "")
                        if link and title and link not in candidates:
                            candidates[link] = {
                                "url": link,
                                "title": title,
                                "summary": entry.get("summary", ""),
                                "source": entry.get("source", {}).get("title", "Google News"),
                                "date": entry.get("published", "")
                            }
                except Exception:
                    continue

        if not candidates:
            return []

        articles = []
        candidate_items = list(candidates.values())[:max_results]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(AdvancedNewsAggregator.extract_text_fallback, item["url"], item["summary"]): item 
                for item in candidate_items
            }
            for future in as_completed(future_map):
                item = future_map[future]
                try:
                    text, quality = future.result()
                    if text:
                        articles.append(Article(
                            title=item["title"],
                            full_text=text[:3000],
                            url=item["url"],
                            source=item["source"],
                            date=item["date"],
                            content_quality=quality
                        ))
                except Exception:
                    continue

        return articles

# =========================================================
# Quantitative Risk Weight Calculations
# =========================================================
def get_source_reliability(source: str) -> float:
    source = (source or "").lower()
    if any(x in source for x in ["central bank", "cbe", "egx", "fra", "regulator", "البنك المركزي", "الهيئة"]): return 1.0
    if any(x in source for x in ["reuters", "bloomberg", "financial times", "wall street journal", "cnbc"]): return 0.9
    if any(x in source for x in ["enterprise", "zawya", "arab finance", "المال", "مصراوي", "اليوم السابع", "الأهرام", "الشروق"]): return 0.8
    return 0.65 if source else 0.55

def calculate_recency_factor(date_string: str) -> float:
    if not date_string: return 0.6
    try:
        dt = parsedate_to_datetime(date_string)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        days_old = max(0, (datetime.now(timezone.utc) - dt).days)
        if days_old <= 7: return 1.00
        if days_old <= 30: return 0.92
        if days_old <= 90: return 0.82
        if days_old <= 180: return 0.72
        if days_old <= 365: return 0.62
        return 0.45
    except Exception:
        return 0.6

def deduplicate_risky_articles(results: List[RiskAnalysisResult]) -> List[RiskAnalysisResult]:
    if len(results) <= 1: 
        return results
    unique_results = []
    seen_titles = []
    for r in sorted(results, key=lambda x: x.score, reverse=True):
        norm_title = ArabicProcessor.normalize(r.article.title)
        # Using fuzz.ratio with a strict 95 threshold to avoid swallowing distinct news articles
        if not any(fuzz.ratio(norm_title, stitle) >= 95 for stitle in seen_titles):
            unique_results.append(r)
            seen_titles.append(norm_title)
    return unique_results

# =========================================================
# High-Intelligence Banking LLM Engine
# =========================================================
class PrecisionBankingLLMEngine:
    DOMAINS_TAXONOMY = {
        "المالية والسيولة": {"severity": 10},
        "التشغيلية وسلاسل الإمداد": {"severity": 8},
        "القانونية والتنظيمية": {"severity": 9},
        "السوق والاقتصاد الكلي": {"severity": 6},
        "السمعة والطوارئ": {"severity": 7}
    }

    NEGATION_PATTERNS = [r"لا\s+صحة", r"ينف[يى]", r"تنف[يى]", r"نفت", r"نفى", r"غير\s+صحيح", r"لم\s+يحدث", r"براءة", r"تسوية\s+ودية"]

    def __init__(self, api_key: str):
        self.api_key = api_key.strip()
        self.domain_names = list(self.DOMAINS_TAXONOMY.keys())

    def _call_groq_api(self, payload: dict) -> dict:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        import random
        for attempt in range(3):
            try:
                # Add random jitter to prevent concurrent workers from hitting the API at the exact same millisecond
                time.sleep(1.5 + random.uniform(0, 1.5))
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if not content:
                        continue
                    content = re.sub(r"^```(?:json)?\s*", "", content)
                    content = re.sub(r"\s*```$", "", content)
                    try:
                        return json.loads(content)
                    except Exception as parse_e:
                        with open("groq_debug.log", "a", encoding="utf-8") as f:
                            f.write(f"Attempt {attempt}: JSON Parse Error - {parse_e}\nRaw Content: {content}\n")
                        continue
                elif response.status_code == 429:
                    wait_match = re.search(r"in ([\d\.]+)s", response.text)
                    if wait_match:
                        time.sleep(float(wait_match.group(1)) + 1.0)
                    else:
                        time.sleep(10)
                else:
                    with open("groq_debug.log", "a", encoding="utf-8") as f:
                        f.write(f"Attempt {attempt}: Status {response.status_code} - {response.text}\n")
            except Exception as req_e:
                with open("groq_debug.log", "a", encoding="utf-8") as f:
                    f.write(f"Attempt {attempt}: Request Exception - {req_e}\n")
                time.sleep(2)
        return None

    def analyze_article(self, article: Article, company_name: str, aliases: List[str]) -> RiskAnalysisResult:
        payload = {
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": f"""You are a Senior Credit Risk Auditor at a major corporate bank.
Your objective is to evaluate whether this news article contains GENUINE MATERIAL RISKS directly threatening the creditworthiness of "{company_name}".

Article Title: {article.title}
Article Text: {article.full_text[:2200]}

STRICT EVALUATION MANDATES:
1. NON-RISK CRITERIA:
   - Corporate expansion, new investments, profit increases, partnerships, routine establishment, or general stock market summaries MUST BE CLASSIFIED AS has_risk = false with events = [].
2. RISK CRITERIA:
   - Consider all material risk events including: loan defaults, liquidity deficits, debt restructuring, lawsuits, international arbitration, production shutdowns, major operational fires, regulatory fines, corruption, fraud, rating downgrades, significant profit declines, severe cost increases, supply chain disruptions, or macroeconomic pressures threatening the company.
3. NEGATIONS:
   - If the company denies the allegation or reaches a resolved settlement with no penalties, has_risk MUST be false.
4. Allowed Categories: {", ".join(self.domain_names)}.

EXAMPLES OF CORRECT BEHAVIOR:
Example 1 (Clear Risk):
Text: "أعلنت الشركة عن خسائر فادحة وتراكم في الديون وتراجع المبيعات"
Output: {{"has_risk": true, "reason": "خسائر وتراكم الديون وتراجع المبيعات يشير إلى خطر مالي وتشغيلي كبير", "events": [{{"domain": "المالية والسيولة", "evidence": "خسائر فادحة وتراكم في الديون", "severity": 8}}]}}

Example 2 (False Alarm / Positive News / Routine):
Text: "الشركة تفتتح فرعاً جديداً وتعلن عن زيادة في الأرباح وإطلاق تطبيق جديد"
Output: {{"has_risk": false, "reason": "أخبار إيجابية وتوسعات روتينية، لا تمثل خطراً ائتمانياً", "events": []}}

Example 3 (Market Pressure Risk):
Text: "الشركة تعاني من ضغوط كبيرة بسبب ارتفاع تكاليف الشحن وضعف سلاسل الإمداد"
Output: {{"has_risk": true, "reason": "ارتفاع التكاليف وضعف سلاسل الإمداد يشكل ضغطاً على العمليات والربحية", "events": [{{"domain": "التشغيلية وسلاسل الإمداد", "evidence": "ارتفاع تكاليف الشحن وضعف سلاسل الإمداد", "severity": 6}}]}}

Output strict JSON:
{{
  "has_risk": true or false,
  "reason": "Clear justification in formal Arabic",
  "events": [
    {{
      "domain": "Exact Domain Name",
      "evidence": "Direct quote sentence demonstrating the risk",
      "severity": 1 to 10
    }}
  ]
}}"""}],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        parsed = self._call_groq_api(payload)
        
        if not parsed:
            parsed = {"has_risk": False, "reason": "تعذر تحليل الخبر بسبب ضغط على الخادم (API Error). يرجى المحاولة لاحقاً.", "events": []}
            
        # Protect against LLM hallucinating has_risk=True but forgetting to provide the events array
        if bool(parsed.get("has_risk", False)) and len(parsed.get("events", [])) == 0:
            parsed["events"] = [{
                "domain": "القانونية والتنظيمية",
                "evidence": parsed.get("reason", "مخاطر محتملة تم رصدها بواسطة الذكاء الاصطناعي"),
                "severity": 6
            }]

        if parsed:
            with open("llm_responses.log", "a", encoding="utf-8") as f:
                f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                
            events = []
            source_rel = get_source_reliability(article.source)
            recency = calculate_recency_factor(article.date)

            for ev in parsed.get("events", []):
                evidence = ev.get("evidence", "").strip()
                if not evidence:
                    continue
                
                domain = ev.get("domain", "")
                if domain not in self.domain_names:
                    domain = "القانونية والتنظيمية" if ("قضا" in domain or "نزاع" in domain or "تحكيم" in domain) else "المالية والسيولة"
                
                severity = min(max(int(ev.get("severity", 6)), 1), 10)
                relevance = 0.95 
                
                combined_context = f"{article.title} {evidence}"
                entity_conf = calculate_entity_confidence(combined_context, company_name, aliases)
                
                contrib = severity * source_rel * entity_conf * recency * relevance * article.content_quality
                
                events.append(RiskEvent(
                    domain_ar=domain, matched_keywords=[], evidence_sentence=evidence,
                    severity=severity, score_contribution=contrib, relevance=relevance,
                    source_reliability=source_rel, entity_confidence=entity_conf, recency_factor=recency
                ))

            is_risky = bool(parsed.get("has_risk", False)) and len(events) > 0

            return RiskAnalysisResult(
                has_risk=is_risky,
                article=article,
                score=sum(e.score_contribution for e in events) if is_risky else 0.0,
                reason=parsed.get("reason", "تم الفحص والتحقق بواسطة المحرك الائتماني الذكي"),
                events=events if is_risky else [],
                confidence=0.95 if is_risky else 0.85
            )

        return RiskAnalysisResult(
            has_risk=False, article=article, score=0.0,
            reason="خبر اعتيادي لا يتضمن مؤشرات مخاطر ائتمانية مؤكدة", events=[], confidence=0.75
        )

    def _is_negated(self, sentence: str) -> bool:
        normalized = ArabicProcessor.normalize(sentence)
        return any(re.search(ArabicProcessor.normalize(p), normalized) for p in self.NEGATION_PATTERNS)

# =========================================================
# CRS & Aggregations
# =========================================================
def calculate_crs(results: List[RiskAnalysisResult]) -> float:
    if not results: return 0.0
    raw_score = sum(event.score_contribution for result in results for event in result.events)
    crs = 100 * (1 - np.exp(-raw_score / 45.0))
    return round(float(min(100, max(0, crs))), 1)

def get_decision(crs: float) -> Tuple[str, str, str]:
    if crs <= 15: return "منخفض", "🟢 متابعة ائتمانية عادية", "success"
    if crs <= 45: return "متوسط", "🟡 مراجعة ائتمانية معززة", "warning"
    return "مرتفع", "🔴 تصعيد فوري ومراجعة ائتمانية تفصيلية", "error"

def aggregate_stats(results: List[RiskAnalysisResult]) -> Dict:
    stats = {d: {"count": 0, "score": 0.0} for d in PrecisionBankingLLMEngine.DOMAINS_TAXONOMY.keys()}
    for result in results:
        for event in result.events:
            if event.domain_ar in stats:
                stats[event.domain_ar]["count"] += 1
                stats[event.domain_ar]["score"] += event.score_contribution
    return stats

# =========================================================
# Streamlit Dashboard UI
# =========================================================
def main():
    st.set_page_config(page_title=APP_NAME, layout="wide")

    with st.sidebar:
        st.header("⚙️ الإعدادات")
        api_key = st.text_input("Groq API Key (مطلوب)", type="password")
        st.markdown("[احصل على المفتاح مجاناً من Groq](https://console.groq.com/keys)")
        st.info("💡 **المعمارية**: تحليل ائتماني متوازي يعتمد على Llama 3.3 70B مع دعم Fallback السريع.")

    st.title("🛡️ RiskPulse AI")
    st.caption("AI-Powered Corporate Credit Risk Intelligence & Early Warning System")

    col1, col2 = st.columns([3, 1])
    with col1:
        company = st.text_input("اسم الشركة:", placeholder="جهينة، ماريدايف، أوراسكوم...")
    with col2:
        max_news = st.slider("عدد الأخبار المستهدفة", 10, 50, 30)

    if st.button("🚀 تحليل شامل", type="primary", use_container_width=True):
        if not api_key.strip():
            st.error("⚠️ يرجى إدخال Groq API Key في القائمة الجانبية للمتابعة.")
            return

        if not company.strip():
            st.error("⚠️ يرجى إدخال اسم الشركة.")
            return

        official_name, aliases = resolve_company(company)
        st.info(f"🏢 الشركة المستهدفة: **{official_name}**")

        with st.spinner("جاري استرجاع الأخبار التاريخية وتجهيز المقالات..."):
            articles = AdvancedNewsAggregator.fetch_news(official_name, aliases, max_results=max_news, max_workers=6)

        if not articles:
            st.warning("تعذر جلب أخبار للشركة في الوقت الحالي.")
            return
        
        st.success(f"تم جلب **{len(articles)}** مقالة. جاري التحليل الائتماني الذكي بالتوازي...")

        engine = PrecisionBankingLLMEngine(api_key=api_key)
        all_results = []
        progress_bar = st.progress(0)
        completed = 0

        # Limit LLM concurrency to 2 to avoid hitting the strict 8000 TPM rate limit instantly
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_map = {executor.submit(engine.analyze_article, art, official_name, aliases): art for art in articles}
            for future in as_completed(future_map):
                try:
                    res = future.result()
                    if res:
                        all_results.append(res)
                except Exception:
                    pass
                completed += 1
                progress_bar.progress(completed / len(articles))

        progress_bar.empty()

        risky = [r for r in all_results if r.has_risk]
        safe = [r for r in all_results if not r.has_risk]
        
        unique_risky = deduplicate_risky_articles(risky)
        total_crs = calculate_crs(unique_risky)
        level, decision, badge = get_decision(total_crs)

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Composite Risk Score", f"{total_crs:.1f}/100")
        m2.metric("Risk Level", level)
        m3.metric("Validated Risk Events", sum(len(r.events) for r in unique_risky))
        m4.metric("Analyzed Articles", len(articles))
        getattr(st, badge)(f"**Decision Support:** {decision}")

        st.markdown("---")
        st.subheader("📊 توزيع المخاطر القطاعية")
        stats = aggregate_stats(unique_risky)
        cols = st.columns(5)
        for idx, (domain, data) in enumerate(stats.items()):
            with cols[idx]:
                st.metric(domain, f"{data['count']} أحداث", f"{data['score']:.1f}")

        st.markdown("---")
        tab1, tab2 = st.tabs([f"🚨 مخاطر مؤكدة ({len(unique_risky)})", f"✅ لا توجد مخاطر ({len(safe)})"])

        with tab1:
            if not unique_risky:
                st.info("لم يتم رصد أي مخاطر مادية مؤكدة على الشركة في هذه العينة.")
            for result in unique_risky:
                with st.expander(f"{result.article.title[:90]} — Score: {result.score:.1f}"):
                    st.write(f"**المصدر:** {result.article.source} | **التاريخ:** {result.article.date}")
                    st.write(f"**التقييم الائتماني:** {result.reason}")
                    st.write(f"**الرابط:** {result.article.url}")
                    for event in result.events:
                        st.markdown(f"### {event.domain_ar}")
                        st.write(f"📍 {event.evidence_sentence}")
                        st.caption(f"Severity: {event.severity}/10 | Relevance: {event.relevance:.2f} | Reliability: {event.source_reliability:.2f} | Entity Conf: {event.entity_confidence:.2f} | Contribution: {event.score_contribution:.2f}")

        with tab2:
            if not safe:
                st.write("جميع المقالات رصدت مخاطر.")
            for result in safe:
                with st.expander(result.article.title[:90]):
                    st.write(f"**المصدر:** {result.article.source}")
                    st.write(f"**السبب:** {result.reason}")
                    st.write(f"**الرابط:** {result.article.url}")

if __name__ == "__main__":
    main()