# app.py - Updated OBPP Hybrid Compliance Backend with multi-guideline support and classification
import os
import uuid
import json
import re
import logging
import shutil
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from openai import OpenAI
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from docx import Document
from tenacity import retry, stop_after_attempt, wait_fixed
from jsonschema import validate as jsonschema_validate

# -------- CONFIG --------
load_dotenv()
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")

MODEL = os.getenv("COMPLIANCE_MODEL", "gpt-4o")
PASS_THRESHOLD = 95.0
WARN_THRESHOLD = 80.0
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
import httpx
client = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client()) if OPENAI_API_KEY else None

# -------- Regex and Heuristics --------
SEBI_REG_REGEX = re.compile(r"(SEBI\s*(Reg(istration)?|Reg\.?)\s*(No|Number|#)?\s*[:\-]?\s*[A-Za-z0-9\-/]+)", re.IGNORECASE)
WARNING_PATTERNS = [
    r"mutual funds are subject to market risk",
    r"investments? are subject to market risk",
    r"returns? are not guaranteed",
    r"निवेश बाज़ार जोखिम के अधीन है",  # Hindi
    r"nivesh bazar jokhim ke adheen hai",  # Hinglish
]
FIXED_RETURN_PATTERNS = [r"guaranteed\s+\d+%?", r"assured\s+returns?", r"fixed\s+returns?"]
EXCHANGE_LOGO_PAT = re.compile(r"(bse|nse|stock exchange|exchange\s+logo)", re.IGNORECASE)
PRODUCT_FIELDS_PAT = re.compile(r"(issuer|tenor|rating|security|YTM|yield to maturity|coupon|maturity)", re.IGNORECASE)
HYPERLINK_PAT = re.compile(r"https?://|www\.", re.IGNORECASE)
SMS_HYPERLINK_PAT = re.compile(r"(sms:|http[s]?:\/\/\S+|bit\.ly|tinyurl|short\.ly)", re.IGNORECASE)
REGIONAL_LANG_PAT = re.compile(r"(Hindi|Bengali|Tamil|Telugu|Marathi|Kannada|Gujarati|Malayalam)", re.IGNORECASE)  # heuristic
CELEBRITY_PATTERNS = [r"brand ambassador", r"celebrity", r"actor", r"actress", r"cricket star", r"film star"]
SUPERLATIVE_PAT = re.compile(r"\b(best|#1|world['’]?s|leading|unrivalled|unbeatable)\b", re.IGNORECASE)
INFLATION_BEAT_PAT = re.compile(r"(beat inflation|beat the inflation|beat the market)", re.IGNORECASE)
DISCOUNT_PROMISE_PAT = re.compile(r"(guarantee|assured|assurance|assured returns?)", re.IGNORECASE)
GAMES_PRIZES_PAT = re.compile(r"(win .* prize|contest|game|lucky draw|prize)", re.IGNORECASE)
CLIENT_DATA_SHARE_PAT = re.compile(r"(share.*client|client data|customer data|personal data).*third", re.IGNORECASE)
LIABILITIES_PAT = re.compile(r"(liability|liabilities|disclaimer|not responsible|no liability)", re.IGNORECASE)
APPROVALS_PAT = re.compile(r"(approved by|approval|template|pre-approved|sanctioned)", re.IGNORECASE)
UNDERTAKING_PAT = re.compile(r"undertaking|we undertake|undertakes", re.IGNORECASE)
EXEMPTION_PAT = re.compile(r"exempt|exemption", re.IGNORECASE)
RETENTION_PAT = re.compile(r"(retain|retention).{0,20}(5\s*years|5y|5 years)", re.IGNORECASE)
REAPPROVAL_PAT = re.compile(r"(re-?approval|reapproval|renewal).{0,50}(180|one hundred eighty|180 days|180days)", re.IGNORECASE)
SUSPENSION_PAT = re.compile(r"(suspend|suspension|suspended)", re.IGNORECASE)
THIRD_PARTY_PAT = re.compile(r"(third[- ]party|vendor|agency|agency action)", re.IGNORECASE)
CLAIMS_SOURCED_PAT = re.compile(r"(source:|according to|as per|study by|survey by|data from)", re.IGNORECASE)
SIMPLE_LANGUAGE_HEURISTIC = lambda s: (sum(len(w) for w in re.findall(r"\w+", s)) / max(1, len(re.findall(r"\w+", s)))) <= 6.5

# -------- Deterministic checks --------
def deterministic_checks(chunk: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    # name/address/reg
    m = SEBI_REG_REGEX.search(chunk)
    out["name_address_reg"] = {"value": bool(m), "confidence": 0.99 if m else 0.0, "evidence": m.group(0) if m else "", "source": "regex"}
    # warnings
    found_warn = None
    for p in WARNING_PATTERNS:
        if re.search(p, chunk, flags=re.IGNORECASE):
            found_warn = re.search(p, chunk, flags=re.IGNORECASE).group(0)
            break
    out["standard_warning_present"] = {"value": bool(found_warn), "confidence": 0.9 if found_warn else 0.0, "evidence": found_warn or "", "source": "regex"}
    # warning font size: not determinable from plain text -> low confidence false
    out["warning_font_size_ok"] = {"value": False, "confidence": 0.0, "evidence": "", "source": "heuristic"}
    # AV duration: can't detect from text alone
    out["av_duration_ok"] = {"value": False, "confidence": 0.0, "evidence": "", "source": "heuristic"}
    # regional languages
    out["regional_languages_used"] = {"value": bool(REGIONAL_LANG_PAT.search(chunk)), "confidence": 0.8 if REGIONAL_LANG_PAT.search(chunk) else 0.0, "evidence": (REGIONAL_LANG_PAT.search(chunk).group(0) if REGIONAL_LANG_PAT.search(chunk) else ""), "source": "regex"}
    # hyperlink for sms
    out["hyperlink_for_sms_ok"] = {"value": bool(HYPERLINK_PAT.search(chunk) or SMS_HYPERLINK_PAT.search(chunk)), "confidence": 0.85 if HYPERLINK_PAT.search(chunk) or SMS_HYPERLINK_PAT.search(chunk) else 0.0, "evidence": (HYPERLINK_PAT.search(chunk).group(0) if HYPERLINK_PAT.search(chunk) else (SMS_HYPERLINK_PAT.search(chunk).group(0) if SMS_HYPERLINK_PAT.search(chunk) else "")), "source": "regex"}
    # product details
    pf = PRODUCT_FIELDS_PAT.search(chunk)
    out["product_details_disclosed"] = {"value": bool(pf), "confidence": 0.85 if pf else 0.0, "evidence": pf.group(0) if pf else "", "source": "regex"}
    # exchange logo absent (we interpret presence of string as forbidden)
    ex = EXCHANGE_LOGO_PAT.search(chunk)
    out["exchange_logo_absent"] = {"value": False if ex else True, "confidence": 0.8 if ex else 0.2, "evidence": ex.group(0) if ex else "", "source": "regex"}
    # claims sourced
    cs = CLAIMS_SOURCED_PAT.search(chunk)
    out["claims_sourced"] = {"value": bool(cs), "confidence": 0.8 if cs else 0.0, "evidence": cs.group(0) if cs else "", "source": "regex"}
    # simple language heuristic
    simple = SIMPLE_LANGUAGE_HEURISTIC(chunk)
    out["simple_language_used"] = {"value": bool(simple), "confidence": 0.6 if simple else 0.5, "evidence": f"avg_word_len_ok={simple}", "source": "heuristic"}
    # fixed returns
    fx = next((re.search(p, chunk, re.IGNORECASE) for p in FIXED_RETURN_PATTERNS if re.search(p, chunk, re.IGNORECASE)), None)
    out["fixed_returns_warning_present"] = {"value": bool(fx), "confidence": 0.95 if fx else 0.0, "evidence": fx.group(0) if fx else "", "source": "regex"}
    # logos without approval - regex can't fully detect; heuristic default false
    out["no_other_logos_without_approval"] = {"value": True, "confidence": 0.4, "evidence": "", "source": "heuristic"}
    # prohibited / tone checks
    out["no_illegal_or_false"] = {"value": True, "confidence": 0.5, "evidence": "", "source": "heuristic"}
    exag = next((re.search(p, chunk, re.IGNORECASE) for p in [r"best ever", r"once in a lifetime", r"miracle"] if re.search(p, chunk, re.IGNORECASE)), None)
    out["no_exaggerated_slogans"] = {"value": False if exag else True, "confidence": 0.9 if exag else 0.7, "evidence": exag.group(0) if exag else "", "source": "regex"}
    sup = SUPERLATIVE_PAT.search(chunk)
    out["no_superlatives_unsubstantiated"] = {"value": False if sup else True, "confidence": 0.9 if sup else 0.7, "evidence": sup.group(0) if sup else "", "source": "regex"}
    infl = INFLATION_BEAT_PAT.search(chunk)
    out["no_inflation_beating_claims"] = {"value": False if infl else True, "confidence": 0.9 if infl else 0.7, "evidence": infl.group(0) if infl else "", "source": "regex"}
    disc = re.search(r"\b(discredit|disparag)", chunk, re.IGNORECASE)
    out["no_discrediting_competitors"] = {"value": False if disc else True, "confidence": 0.85 if disc else 0.7, "evidence": disc.group(0) if disc else "", "source": "regex"}
    cele = next((re.search(p, chunk, re.IGNORECASE) for p in CELEBRITY_PATTERNS if re.search(p, chunk, re.IGNORECASE)), None)
    out["no_celebrities"] = {"value": False if cele else True, "confidence": 0.9 if cele else 0.7, "evidence": cele.group(0) if cele else "", "source": "regex"}
    assured = DISCOUNT_PROMISE_PAT.search(chunk)
    out["no_assured_returns"] = {"value": False if assured else True, "confidence": 0.9 if assured else 0.7, "evidence": assured.group(0) if assured else "", "source": "regex"}
    sebi_logo = re.search(r"SEBI\s+logo", chunk, re.IGNORECASE)
    out["no_sebi_logo"] = {"value": False if sebi_logo else True, "confidence": 0.9 if sebi_logo else 0.7, "evidence": sebi_logo.group(0) if sebi_logo else "", "source": "regex"}
    # other compliances
    out["approvals_required_or_template"] = {"value": bool(APPROVALS_PAT.search(chunk)), "confidence": 0.8 if APPROVALS_PAT.search(chunk) else 0.0, "evidence": (APPROVALS_PAT.search(chunk).group(0) if APPROVALS_PAT.search(chunk) else ""), "source": "regex"}
    out["undertakings_provided"] = {"value": bool(UNDERTAKING_PAT.search(chunk)), "confidence": 0.8 if UNDERTAKING_PAT.search(chunk) else 0.0, "evidence": (UNDERTAKING_PAT.search(chunk).group(0) if UNDERTAKING_PAT.search(chunk) else ""), "source": "regex"}
    out["exemptions_applied_correctly"] = {"value": bool(EXEMPTION_PAT.search(chunk)), "confidence": 0.6 if EXEMPTION_PAT.search(chunk) else 0.0, "evidence": (EXEMPTION_PAT.search(chunk).group(0) if EXEMPTION_PAT.search(chunk) else ""), "source": "regex"}
    out["quarterly_upload_done"] = {"value": False, "confidence": 0.0, "evidence": "", "source": "heuristic"}
    gp = GAMES_PRIZES_PAT.search(chunk)
    out["no_games_or_prizes"] = {"value": False if gp else True, "confidence": 0.9 if gp else 0.7, "evidence": gp.group(0) if gp else "", "source": "regex"}
    out["retention_5y"] = {"value": bool(RETENTION_PAT.search(chunk)), "confidence": 0.8 if RETENTION_PAT.search(chunk) else 0.0, "evidence": (RETENTION_PAT.search(chunk).group(0) if RETENTION_PAT.search(chunk) else ""), "source": "regex"}
    out["reapprovals_after_180d"] = {"value": bool(REAPPROVAL_PAT.search(chunk)), "confidence": 0.7 if REAPPROVAL_PAT.search(chunk) else 0.0, "evidence": (REAPPROVAL_PAT.search(chunk).group(0) if REAPPROVAL_PAT.search(chunk) else ""), "source": "regex"}
    out["medium_changes_ok"] = {"value": True, "confidence": 0.5, "evidence": "", "source": "heuristic"}
    out["suspension_rules_followed"] = {"value": bool(SUSPENSION_PAT.search(chunk)), "confidence": 0.6 if SUSPENSION_PAT.search(chunk) else 0.0, "evidence": (SUSPENSION_PAT.search(chunk).group(0) if SUSPENSION_PAT.search(chunk) else ""), "source": "regex"}
    out["third_party_action_compliant"] = {"value": False if THIRD_PARTY_PAT.search(chunk) else True, "confidence": 0.7 if THIRD_PARTY_PAT.search(chunk) else 0.4, "evidence": (THIRD_PARTY_PAT.search(chunk).group(0) if THIRD_PARTY_PAT.search(chunk) else ""), "source": "regex"}
    out["no_client_data_sharing"] = {"value": not bool(CLIENT_DATA_SHARE_PAT.search(chunk)), "confidence": 0.85 if CLIENT_DATA_SHARE_PAT.search(chunk) else 0.4, "evidence": CLIENT_DATA_SHARE_PAT.search(chunk).group(0) if CLIENT_DATA_SHARE_PAT.search(chunk) else "", "source": "regex"}
    out["liabilities_disclaimed"] = {"value": bool(LIABILITIES_PAT.search(chunk)), "confidence": 0.8 if LIABILITIES_PAT.search(chunk) else 0.0, "evidence": (LIABILITIES_PAT.search(chunk).group(0) if LIABILITIES_PAT.search(chunk) else ""), "source": "regex"}
    out["penalty_awareness"] = {"value": True, "confidence": 0.3, "evidence": "", "source": "heuristic"}
    # Add for accurate_info and is_advertisement if needed
    out["accurate_info"] = {"value": True, "confidence": 0.5, "evidence": "", "source": "heuristic"}
    out["is_advertisement"] = {"value": True, "confidence": 0.5, "evidence": "", "source": "heuristic"}
    return out

# -------- OBPP Categories and Guidance --------
OBPP_CATEGORIES = {
    "Forms of Communication": ["is_advertisement"],
    "Disclosures": [
        "name_address_reg","accurate_info","standard_warning_present","warning_font_size_ok",
        "av_duration_ok","regional_languages_used","hyperlink_for_sms_ok",
        "product_details_disclosed","exchange_logo_absent","claims_sourced",
        "simple_language_used","fixed_returns_warning_present","no_other_logos_without_approval"
    ],
    "Prohibitions": [
        "no_illegal_or_false","no_exaggerated_slogans","no_superlatives_unsubstantiated",
        "no_inflation_beating_claims","no_discrediting_competitors","no_celebrities",
        "no_assured_returns","no_sebi_logo"
    ],
    "Other Compliances": [
        "approvals_required_or_template","undertakings_provided","exemptions_applied_correctly",
        "quarterly_upload_done","no_games_or_prizes","retention_5y","reapprovals_after_180d",
        "medium_changes_ok","suspension_rules_followed","third_party_action_compliant",
        "no_client_data_sharing","liabilities_disclaimed"
    ],
    "Penalties": ["penalty_awareness"]
}

GUIDANCE_MAP = {
    # Form / identity
    "is_advertisement": {
        "pass": "This content is clearly an advertisement (format & intent recognized).",
        "fail": "Ensure the communication mode is labeled and consistent (e.g., 'Advertisement' header or appropriate metadata)."
    },
    "name_address_reg": {
        "pass": "SEBI registration number and required issuer identity are clearly present.",
        "fail": "Add the intermediary's name, address and SEBI Registration No. (e.g., 'SEBI Reg. No: INZ00012345') in a readable location."
    },
    "accurate_info": {
        "pass": "Key facts (rates, numbers, names) appear consistent and plausible.",
        "fail": "Verify and correct any factual inaccuracies (dates, percentages, product names) before publishing."
    },
    # Warnings / disclosures
    "standard_warning_present": {
        "pass": "Standard risk disclaimer present (e.g., 'Mutual funds are subject to market risk').",
        "fail": "Insert the mandatory risk disclaimer; recommended phrasing: 'Mutual funds are subject to market risk. Please read the offer document carefully.'"
    },
    "warning_font_size_ok": {
        "pass": "Warning/disclaimer text appears to meet prominence/readability requirements.",
        "fail": "Increase the disclaimer font size and prominence so it is legible and not visually de-emphasized (follow brand/OBPP specs)."
    },
    "av_duration_ok": {
        "pass": "AV disclaimers meet duration/word-count thresholds (e.g., visible/readable for required seconds).",
        "fail": "Ensure AV disclaimers are displayed for the regulator-prescribed duration and readable when spoken/displayed."
    },
    "regional_languages_used": {
        "pass": "Regional language warnings provided as required.",
        "fail": "Provide region-appropriate language versions of the mandatory warnings or a clear link to the translation."
    },
    "hyperlink_for_sms_ok": {
        "pass": "Hyperlinks (or SMS short links) to full terms/offer documents are present and valid.",
        "fail": "Add a valid hyperlink or short URL (SMS-compliant) to the full offer document or disclosures."
    },
    "product_details_disclosed": {
        "pass": "Product-level details (issuer, tenor, rating, YTM/coupon) are disclosed.",
        "fail": "Disclose required product details — issuer, tenor, rating, yield-to-maturity or equivalent fields."
    },
    "exchange_logo_absent": {
        "pass": "No stock-exchange logo wrongly used in the creative (no misleading affiliation).",
        "fail": "Remove exchange logos or obtain explicit authorization; do not imply listing/endorsement by an exchange."
    },
    "claims_sourced": {
        "pass": "Claims include sources (surveys, data) where applicable.",
        "fail": "Add verifiable sources for claims (e.g., 'Source: XYZ survey, 2024') or remove unsupported claims."
    },
    "simple_language_used": {
        "pass": "Language is plain and accessible to retail investors.",
        "fail": "Rewrite complex sentences into simple, plain-language statements targeted at the retail investor."
    },
    "fixed_returns_warning_present": {
        "pass": "If fixed/assured return language used, an explicit warning and context exists.",
        "fail": "Remove any claim of 'fixed' or 'assured' returns, or explicitly qualify and back with permitted wording and disclosures."
    },
    "no_other_logos_without_approval": {
        "pass": "No third-party logos used without explicit approval.",
        "fail": "Remove or obtain approvals for third-party logos shown in the creative."
    },
    # Prohibitions & tone
    "no_illegal_or_false": {
        "pass": "No illegal or false statements detected.",
        "fail": "Remove statements that are false, illegal or misleading and re-run compliance checks."
    },
    "no_exaggerated_slogans": {
        "pass": "No exaggerated slogans (e.g., 'best ever') detected.",
        "fail": "Remove exaggerated marketing slogans; use objective, verifiable phrasing instead."
    },
    "no_superlatives_unsubstantiated": {
        "pass": "No unsubstantiated superlatives (#1, 'leading') present.",
        "fail": "Remove superlatives or add evidence/sources to substantiate any ranking claims."
    },
    "no_inflation_beating_claims": {
        "pass": "No claims of 'beat inflation' or similar present.",
        "fail": "Remove or substantiate any claim that promises to 'beat inflation' — these are sensitive and must be supported with evidence."
    },
    "no_discrediting_competitors": {
        "pass": "No discrediting of competitors detected.",
        "fail": "Do not disparage competitors; remove comparative language that discredits other firms/products."
    },
    "no_celebrities": {
        "pass": "No celebrity endorsement detected (or authorizations exist).",
        "fail": "Remove celebrity imagery/text unless documented approvals and disclosures are attached."
    },
    "no_assured_returns": {
        "pass": "No assured / guaranteed returns claimed.",
        "fail": "Delete statements implying assured/guaranteed returns; ensure disclaimers are present if any returns are discussed."
    },
    "no_sebi_logo": {
        "pass": "SEBI logo is not used incorrectly.",
        "fail": "Remove any misuse of SEBI's name/logo or obtain explicit permission."
    },
    # Other compliances
    "approvals_required_or_template": {
        "pass": "Approvals and templates used where required.",
        "fail": "Follow the mandated approval flow — seek prior approvals per OBPP before publishing."
    },
    "undertakings_provided": {
        "pass": "Required undertakings are included.",
        "fail": "Add required undertakings and declarations from the responsible signatory."
    },
    "exemptions_applied_correctly": {
        "pass": "Any exemptions are clearly documented and justified.",
        "fail": "Document any claimed exemptions and ensure they are valid under the OBPP rules."
    },
    "quarterly_upload_done": {
        "pass": "Quarterly uploads / records are maintained as required.",
        "fail": "Ensure the campaign is recorded/uploaded in the quarterly compliance registry."
    },
    "no_games_or_prizes": {
        "pass": "No games or prizes included (or approvals exist).",
        "fail": "Remove contests/games/prize mechanics unless specifically allowed and approved."
    },
    "retention_5y": {
        "pass": "Retention policy (5 years) is adhered to / documented.",
        "fail": "Retain campaign artifacts for 5 years per OBPP; document retention location."
    },
    "reapprovals_after_180d": {
        "pass": "Re-approval workflow adhered after 180 days where required.",
        "fail": "If >180 days since approval, re-seek approvals per policy."
    },
    "medium_changes_ok": {
        "pass": "Medium-specific change rules respected (minor edits vs. re-approval).",
        "fail": "Major changes to creative/medium may require fresh approvals — confirm and re-submit if needed."
    },
    "suspension_rules_followed": {
        "pass": "Suspension/withdrawal requirements followed where applicable.",
        "fail": "Follow the suspension/withdrawal procedures if content is non-compliant post-deployment."
    },
    "third_party_action_compliant": {
        "pass": "Third-party vendor/agency actions are compliant and documented.",
        "fail": "Obtain compliance confirmations from third-party vendors/agencies and document them."
    },
    "no_client_data_sharing": {
        "pass": "No unauthorized client data sharing detected.",
        "fail": "Remove or secure any flow that shares client data with third parties; follow privacy rules."
    },
    "liabilities_disclaimed": {
        "pass": "Liability/disclaimer language present and adequate.",
        "fail": "Add clear liability and disclaimer statements as per OBPP templates."
    },
    "penalty_awareness": {
        "pass": "Penalty provisions acknowledged and internal controls exist.",
        "fail": "Ensure the team documents penalty awareness and internal controls to avoid breaches."
    }
}

# -------- Guideline Configurations --------
# Universal ASCI
ASCI_CODE = 'asci'
ASCI_NAME = 'ASCI (Advertising Standards Council of India)'
ASCI_CATEGORIES = {
    "Truthfulness and Honesty": ["is_truthful", "not_misleading", "claims_substantiated"],
    "Decency and Non-Offensiveness": ["decent_language", "not_exploiting_vulnerability"],
    "Fairness": ["fair_competition", "no_disparagement"],
}
ASCI_FIELD_DESCS = {
    'is_truthful': 'Is the content truthful and honest?',
    'not_misleading': 'Does the ad avoid misleading by omission, ambiguity, or exaggeration?',
    'claims_substantiated': 'Are all claims backed by evidence or sources?',
    'decent_language': 'Is the language decent, not obscene or offensive?',
    'not_exploiting_vulnerability': 'Does it avoid exploiting fear, superstition, or vulnerability?',
    'fair_competition': 'Does it promote fair competition without unfair advantage?',
    'no_disparagement': 'Does it avoid disparaging competitors or their products?',
}
ASCI_REGEX_FUNCS = {
    'is_truthful': lambda c: {"value": True, "confidence": 0.5, "evidence": "", "source": "heuristic"},  # LLM heavy
    'not_misleading': lambda c: {"value": not bool(re.search(r"\b(guarantee|assured|100%|risk free)\b", c, re.IGNORECASE)), "confidence": 0.8, "evidence": re.search(r"\b(guarantee|assured|100%|risk free)\b", c, re.IGNORECASE).group(0) if re.search(r"\b(guarantee|assured|100%|risk free)\b", c, re.IGNORECASE) else "", "source": "regex"},
    'claims_substantiated': lambda c: {"value": bool(CLAIMS_SOURCED_PAT.search(c)), "confidence": 0.8, "evidence": CLAIMS_SOURCED_PAT.search(c).group(0) if CLAIMS_SOURCED_PAT.search(c) else "", "source": "regex"},
    'decent_language': lambda c: {"value": not bool(re.search(r"\b(fuck|shit|damn|hell)\b", c, re.IGNORECASE)), "confidence": 0.7, "evidence": "", "source": "regex"},  # Example offensive words
    'not_exploiting_vulnerability': lambda c: {"value": not bool(re.search(r"\b(fear|scare|urgent|limited time)\b", c, re.IGNORECASE)), "confidence": 0.6, "evidence": "", "source": "regex"},
    'fair_competition': lambda c: {"value": True, "confidence": 0.5, "evidence": "", "source": "heuristic"},
    'no_disparagement': lambda c: {"value": not bool(re.search(r"\b(worse|bad|inferior|competitor)\b", c, re.IGNORECASE)), "confidence": 0.7, "evidence": "", "source": "regex"},
}
ASCI_GUIDANCE = {
    'is_truthful': {'pass': 'Content is truthful and honest.', 'fail': 'Ensure all statements are accurate, verifiable, and not deceptive.'},
    'not_misleading': {'pass': 'No misleading elements detected.', 'fail': 'Remove or clarify ambiguous, exaggerated, or omitted information to avoid misleading consumers.'},
    'claims_substantiated': {'pass': 'Claims are substantiated with sources.', 'fail': 'Provide evidence or sources for all claims made in the content.'},
    'decent_language': {'pass': 'Language is decent and appropriate.', 'fail': 'Remove offensive, obscene, or indecent language.'},
    'not_exploiting_vulnerability': {'pass': 'No exploitation of vulnerabilities.', 'fail': 'Avoid content that exploits fear, superstition, or consumer vulnerabilities.'},
    'fair_competition': {'pass': 'Promotes fair competition.', 'fail': 'Ensure the content does not take unfair advantage or mislead about competitors.'},
    'no_disparagement': {'pass': 'No disparagement of competitors.', 'fail': 'Remove any language that disparages or denigrates competitors or their products.'},
}

# AMFI for Mutual Funds
AMFI_CODE = 'amfi'
AMFI_NAME = 'AMFI (Mutual Funds)'
AMFI_CATEGORIES = {
    "Disclosures": ["arn_present", "risk_disclaimer_present", "scheme_details_disclosed", "past_performance_caveat"],
    "Prohibitions": ["no_assured_returns", "no_misleading_performance", "no_celebrity_endorsement"],
    "Other Compliances": ["simple_language", "approval_obtained"],
}
AMFI_FIELD_DESCS = {
    'arn_present': 'Is the AMFI Registration Number (ARN) present?',
    'risk_disclaimer_present': 'Is the standard risk disclaimer present (e.g., "Mutual Funds are subject to market risks...")?',
    'scheme_details_disclosed': 'Are scheme details (name, objective, asset allocation) disclosed?',
    'past_performance_caveat': 'Is past performance disclosed with caveat that it does not guarantee future returns?',
    'no_assured_returns': 'No assured or guaranteed returns claimed?',
    'no_misleading_performance': 'No misleading presentation of performance data?',
    'no_celebrity_endorsement': 'No unauthorized celebrity endorsements?',
    'simple_language': 'Is simple language used for investor understanding?',
    'approval_obtained': 'Evidence of prior approval where required?',
}
AMFI_REGEX_FUNCS = {
    'arn_present': lambda c: {"value": bool(re.search(r"\b(ARN|AMFI Reg)\s*[-: ]?\s*\d+\b", c, re.IGNORECASE)), "confidence": 0.9, "evidence": "", "source": "regex"},
    'risk_disclaimer_present': lambda c: {"value": bool(re.search(r"mutual funds? are subject to market risk", c, re.IGNORECASE)), "confidence": 0.95, "evidence": "", "source": "regex"},
    'scheme_details_disclosed': lambda c: {"value": bool(re.search(r"\b(scheme|fund|objective|allocation|NAV)\b", c, re.IGNORECASE)), "confidence": 0.8, "evidence": "", "source": "regex"},
    'past_performance_caveat': lambda c: {"value": bool(re.search(r"past performance may or may not be sustained", c, re.IGNORECASE)), "confidence": 0.9, "evidence": "", "source": "regex"},
    'no_assured_returns': lambda c: {"value": not bool(re.search(r"\b(assured|guaranteed) returns?\b", c, re.IGNORECASE)), "confidence": 0.9, "evidence": "", "source": "regex"},
    'no_misleading_performance': lambda c: {"value": True, "confidence": 0.5, "evidence": "", "source": "heuristic"},
    'no_celebrity_endorsement': lambda c: {"value": not bool(re.search(r"\b(celebrity|endorsed by|ambassador)\b", c, re.IGNORECASE)), "confidence": 0.8, "evidence": "", "source": "regex"},
    'simple_language': lambda c: {"value": SIMPLE_LANGUAGE_HEURISTIC(c), "confidence": 0.6, "evidence": "", "source": "heuristic"},
    'approval_obtained': lambda c: {"value": bool(re.search(r"\b(approved|AMFI approved)\b", c, re.IGNORECASE)), "confidence": 0.7, "evidence": "", "source": "regex"},
}
AMFI_GUIDANCE = {
    'arn_present': {'pass': 'AMFI ARN is present.', 'fail': 'Include the AMFI Registration Number (ARN) in the content.'},
    'risk_disclaimer_present': {'pass': 'Risk disclaimer is present.', 'fail': 'Add the standard disclaimer: "Mutual Funds are subject to market risks, read all scheme related documents carefully."'},
    'scheme_details_disclosed': {'pass': 'Scheme details are disclosed.', 'fail': 'Disclose scheme name, objective, asset allocation, and other required details.'},
    'past_performance_caveat': {'pass': 'Past performance caveat included.', 'fail': 'Add caveat: "Past performance may or may not be sustained in future."'},
    'no_assured_returns': {'pass': 'No assured returns claimed.', 'fail': 'Remove any claims of assured or guaranteed returns.'},
    'no_misleading_performance': {'pass': 'Performance data not misleading.', 'fail': 'Ensure performance data is presented fairly and not misleading.'},
    'no_celebrity_endorsement': {'pass': 'No celebrity endorsements.', 'fail': 'Remove celebrity endorsements unless authorized.'},
    'simple_language': {'pass': 'Simple language used.', 'fail': 'Use simple, investor-friendly language.'},
    'approval_obtained': {'pass': 'Approval evidence present.', 'fail': 'Obtain and indicate prior approval from AMFI/SEBI where required.'},
}

# Stock Market (NSE/BSE/MCA) - OBPP
STOCK_CODE = 'stock'
STOCK_NAME = 'NSE/BSE/MCA (Stock Market)'
STOCK_CATEGORIES = OBPP_CATEGORIES
STOCK_FIELD_DESCS = {  # Add descriptions for prompt
    "is_advertisement": 'Is this an advertisement?',
    "name_address_reg": 'Is name, address, and SEBI registration present?',
    "accurate_info": 'Is information accurate?',
    "standard_warning_present": 'Is standard warning present?',
    "warning_font_size_ok": 'Is warning font size adequate?',
    "av_duration_ok": 'Is AV duration compliant?',
    "regional_languages_used": 'Are regional languages used where required?',
    "hyperlink_for_sms_ok": 'Is hyperlink for SMS compliant?',
    "product_details_disclosed": 'Are product details disclosed?',
    "exchange_logo_absent": 'Is exchange logo absent or authorized?',
    "claims_sourced": 'Are claims sourced?',
    "simple_language_used": 'Is simple language used?',
    "fixed_returns_warning_present": 'Is fixed returns warning present if applicable?',
    "no_other_logos_without_approval": 'No unauthorized logos?',
    "no_illegal_or_false": 'No illegal or false statements?',
    "no_exaggerated_slogans": 'No exaggerated slogans?',
    "no_superlatives_unsubstantiated": 'No unsubstantiated superlatives?',
    "no_inflation_beating_claims": 'No inflation-beating claims?',
    "no_discrediting_competitors": 'No discrediting competitors?',
    "no_celebrities": 'No unauthorized celebrities?',
    "no_assured_returns": 'No assured returns?',
    "no_sebi_logo": 'No unauthorized SEBI logo?',
    "approvals_required_or_template": 'Approvals or templates used?',
    "undertakings_provided": 'Undertakings provided?',
    "exemptions_applied_correctly": 'Exemptions applied correctly?',
    "quarterly_upload_done": 'Quarterly upload done?',
    "no_games_or_prizes": 'No games or prizes?',
    "retention_5y": 'Retention for 5 years?',
    "reapprovals_after_180d": 'Reapprovals after 180 days?',
    "medium_changes_ok": 'Medium changes compliant?',
    "suspension_rules_followed": 'Suspension rules followed?',
    "third_party_action_compliant": 'Third-party actions compliant?',
    "no_client_data_sharing": 'No unauthorized client data sharing?',
    "liabilities_disclaimed": 'Liabilities disclaimed?',
    "penalty_awareness": 'Penalty awareness?',
}
STOCK_REGEX_FUNCS = {
    f: lambda c, f=f: deterministic_checks(c).get(f, {"value": False, "confidence": 0.0, "evidence": "", "source": "heuristic"}) for f in sum(OBPP_CATEGORIES.values(), [])
}
STOCK_GUIDANCE = GUIDANCE_MAP

# Map of guideline codes to configs
GUIDELINE_CONFIGS = {
    ASCI_CODE: {'name': ASCI_NAME, 'categories': ASCI_CATEGORIES, 'field_descs': ASCI_FIELD_DESCS, 'regex_funcs': ASCI_REGEX_FUNCS, 'guidance': ASCI_GUIDANCE},
    AMFI_CODE: {'name': AMFI_NAME, 'categories': AMFI_CATEGORIES, 'field_descs': AMFI_FIELD_DESCS, 'regex_funcs': AMFI_REGEX_FUNCS, 'guidance': AMFI_GUIDANCE},
    STOCK_CODE: {'name': STOCK_NAME, 'categories': STOCK_CATEGORIES, 'field_descs': STOCK_FIELD_DESCS, 'regex_funcs': STOCK_REGEX_FUNCS, 'guidance': STOCK_GUIDANCE},
}

# -------- Helper functions --------
def extract_text_from_file(file_path: str) -> str:
    if not file_path or not os.path.exists(file_path):
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            reader = PdfReader(file_path)
            pages = []
            for p in reader.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
            return "\n\n".join(pages)
        elif ext == ".docx":
            doc = Document(file_path)
            return "\n\n".join(p.text for p in doc.paragraphs)
        # elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        #     return pytesseract.image_to_string(Image.open(file_path))
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        logging.exception("Failed to extract text from %s", file_path)
        return ""

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    if not text:
        return [""]
    words = re.findall(r"\S+", text)
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]).strip())
        start += chunk_size - overlap
    return chunks or [text]

# -------- SCHEMA Builder --------
def build_schema(selected_guidelines: List[str]) -> Dict:
    properties = {}
    for g in selected_guidelines:
        config = GUIDELINE_CONFIGS[g]
        for f in sum(config['categories'].values(), []):
            prefixed = f"{g}_{f}"
            properties[prefixed] = {
                "type": "object",
                "properties": {
                    "value": {"type": "boolean"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"}
                },
                "required": ["value", "confidence", "evidence"],
                "additionalProperties": False
            }
    return {
        "name": "CompliancePerception",
        "schema": {
            "type": "object",
            "properties": {
                "is_advertisement": {"type": "boolean"},
                "detected_items": {"type": "object", "properties": properties, "additionalProperties": False},
                "improvements": {"type": "array", "items": {"type": "string"}},
                "anomalies": {"type": "array", "items": {"type": "string"}},
                "what_is_right": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["is_advertisement", "detected_items", "improvements", "anomalies", "what_is_right"],
            "additionalProperties": False
        },
        "strict": True
    }

# -------- Prompt Builder --------
def build_sys_prompt(selected_guidelines: List[str]) -> str:
    prompt = "You are a compliance perceiver for Indian financial advertising guidelines. For the input TEXT, return JSON matching the schema. "
    prompt += "Evaluate each field based on the following descriptions:\n"
    for g in selected_guidelines:
        config = GUIDELINE_CONFIGS[g]
        prompt += f"\nGuideline: {config['name']}\n"
        for f, desc in config['field_descs'].items():
            prefixed = f"{g}_{f}"
            prompt += f"{prefixed}: {desc}\n"
    prompt += "\nEach detected field must be {value: bool, confidence: 0-1, evidence: string}."
    return prompt

# -------- LLM Call --------
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def llm_json_call(chunk: str, selected_guidelines: List[str]) -> Dict[str, Any]:
    if not client:
        return {"is_advertisement": True, "detected_items": {}}
    schema = build_schema(selected_guidelines)
    sys_prompt = build_sys_prompt(selected_guidelines)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": chunk}],
            temperature=0.0,
            response_format={"type": "json_schema", "json_schema": schema},
            max_tokens=1500,
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        jsonschema_validate(parsed, schema["schema"])
        return parsed
    except Exception as e:
        logging.warning("LLM json_schema call failed: %s", e)
        # Fallback
        try:
            resp2 = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": chunk}],
                temperature=0.0,
                max_tokens=1500,
            )
            raw2 = resp2.choices[0].message.content
            start = raw2.find("{")
            end = raw2.rfind("}")
            if start != -1 and end != -1:
                candidate = raw2[start:end+1]
                parsed2 = json.loads(candidate)
                return parsed2
        except Exception:
            logging.exception("LLM fallback failed")
    return {"is_advertisement": True, "detected_items": {}}

# -------- Deterministic for all --------
def get_deterministic(selected_guidelines: List[str], chunk: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for g in selected_guidelines:
        config = GUIDELINE_CONFIGS[g]
        for f in sum(config['categories'].values(), []):
            prefixed = f"{g}_{f}"
            func = config['regex_funcs'].get(f, lambda c: {"value": False, "confidence": 0.0, "evidence": "", "source": "heuristic"})
            out[prefixed] = func(chunk)
    return out

# -------- Merge --------
def merge_perceptions(deterministic: Dict[str, Dict[str, Any]], llm_perc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    merged = {}
    llm_items = llm_perc.get("detected_items", {})
    for k, det in deterministic.items():
        llm = llm_items.get(k)
        if llm and 'value' in llm:
            l = {"value": bool(llm['value']), "confidence": float(llm['confidence']), "evidence": str(llm['evidence']), "source": "llm"}
        else:
            l = None
        if det['confidence'] >= 0.9:
            merged[k] = det
        elif l and l['confidence'] > det['confidence']:
            merged[k] = l
        else:
            merged[k] = det
    return merged

# -------- Build per guideline --------
def build_guideline_eval(g_code: str, final: Dict[str, Dict[str, Any]]) -> Dict:
    config = GUIDELINE_CONFIGS[g_code]
    categories = []
    weighted_total = 0.0
    weighted_scored = 0.0
    category_weights = {cat: 1.0 / len(config['categories']) for cat in config['categories']}  # Equal for simplicity
    what_is_right, improvements, anomalies = [], [], []
    for cat, fields in config['categories'].items():
        total = len(fields)
        passed = 0
        sub_criteria = []
        for f in fields:
            prefixed = f"{g_code}_{f}"
            field_obj = final.get(prefixed, {"value": False, "confidence": 0.0, "evidence": "", "source": "none"})
            val = field_obj['value']
            evidence = field_obj['evidence']
            if val:
                passed += 1
                pos = config['guidance'].get(f, {'pass': ''})['pass']
                what_is_right.append(f"{pos} (evidence: {evidence})" if evidence else pos)
            else:
                suggested = config['guidance'].get(f, {'fail': ''})['fail']
                improvements.append(suggested)
                if 'no_' in f or 'not_' in f:  # Heuristic for anomalies
                    anomalies.append(f"Potential violation in {cat}: {f} (evidence: {evidence}).")
            sub_criteria.append({
                "name": f,
                "pass_fail": "Pass" if val else "Fail",
                "confidence": round(field_obj['confidence'], 2),
                "evidence": evidence
            })
        pct = round(100.0 * passed / max(1, total), 2)
        status = "Pass" if pct >= PASS_THRESHOLD else "Warning" if pct >= WARN_THRESHOLD else "Fail"
        cat_weight = category_weights.get(cat, 0.1)
        weighted_total += cat_weight * 100
        weighted_scored += cat_weight * pct
        categories.append({
            "category": cat,
            "category_percentage": pct,
            "status": status,
            "sub_criteria": sub_criteria
        })
    guideline_pct = round(weighted_scored / max(1, weighted_total) * 100, 2)
    return {
        "guideline": config['name'],
        "guideline_percentage": guideline_pct,
        "status": "Pass" if guideline_pct >= PASS_THRESHOLD else "Warning" if guideline_pct >= WARN_THRESHOLD else "Fail",
        "categories": categories,
        "what_is_right": what_is_right,
        "improvements": improvements,
        "anomalies": anomalies
    }

# -------- Aggregate --------
def aggregate_results(guideline_evals: List[Dict]) -> Dict:
    overall_pct = round(sum(e['guideline_percentage'] for e in guideline_evals) / len(guideline_evals), 2) if guideline_evals else 100.0
    overall_status = "Pass" if overall_pct >= PASS_THRESHOLD else "Warning" if overall_pct >= WARN_THRESHOLD else "Fail"
    what_is_right = sum((e['what_is_right'] for e in guideline_evals), [])
    improvements = sum((e['improvements'] for e in guideline_evals), [])
    anomalies_detected = sum((e['anomalies'] for e in guideline_evals), [])
    return {
        "overall_accuracy_percentage": overall_pct,
        "overall_status": overall_status,
        "evaluations": guideline_evals,  # List of guideline dicts with categories
        "what_is_right": list(dict.fromkeys(what_is_right)),
        "improvements": list(dict.fromkeys(improvements)),
        "anomalies_detected": list(dict.fromkeys(anomalies_detected))
    }

# -------- Main compliance flow --------
def check_compliance(file_paths: List[str], selected_guidelines: List[str]) -> List[Dict]:
    results = []
    for path in file_paths:
        text = extract_text_from_file(path)
        chunks = chunk_text(text)
        merged_per_chunk = []
        for ch in chunks:
            det = get_deterministic(selected_guidelines, ch)
            llm_perc = llm_json_call(ch, selected_guidelines) if client else {"detected_items": {}}
            merged = merge_perceptions(det, llm_perc)
            merged_per_chunk.append(merged)
        # Aggregate max confidence
        final = {}
        for chunk_map in merged_per_chunk:
            for k, v in chunk_map.items():
                if k not in final or v['confidence'] > final[k]['confidence']:
                    final[k] = v
        # Build per guideline
        guideline_evals = []
        for g in selected_guidelines:
            eval_g = build_guideline_eval(g, final)
            guideline_evals.append(eval_g)
        agg = aggregate_results(guideline_evals)
        results.append(agg)
    return results

# -------- Classification --------
CLASSIFICATION_SCHEMA = {
    "name": "Classification",
    "schema": {
        "type": "object",
        "properties": {
            "detected_type": {"type": "string", "enum": ["mutual_fund", "investing", "trading", "ipo", "fno_derivatives", "other"]}
        },
        "required": ["detected_type"],
        "additionalProperties": False
    },
    "strict": True
}

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def llm_classify(chunk: str) -> str:
    if not client:
        return "other"
    sys_prompt = "Classify the financial advertisement text as one of: mutual_fund (mutual funds), investing (general investing), trading (stock trading), ipo (IPO related), fno_derivatives (futures, options, derivatives), other. Return JSON {'detected_type': 'mutual_fund'}"
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": chunk}],
            temperature=0.0,
            response_format={"type": "json_schema", "json_schema": CLASSIFICATION_SCHEMA},
            max_tokens=100,
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        return parsed['detected_type']
    except Exception:
        return "other"

def classify_text(text: str) -> str:
    chunks = chunk_text(text, 200)
    types = [llm_classify(ch) for ch in chunks]
    # Majority vote
    from collections import Counter
    count = Counter(types)
    return count.most_common(1)[0][0] if count else "other"

# -------- FastAPI endpoints --------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logging.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": str(exc)}, headers={"Access-Control-Allow-Origin": "*"})

@app.post("/classify-text")
async def classify_endpoint(file: UploadFile = File(None), text: str = Form(None)):
    temp_path = None
    try:
        if file:
            ext = os.path.splitext(file.filename)[1] or ".txt"
            temp_path = f"/tmp/{uuid.uuid4()}{ext}"
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            extracted = extract_text_from_file(temp_path)
        elif text:
            extracted = text
        else:
            return {"detected_type": "other"}
        detected = classify_text(extracted)
        return {"detected_type": detected}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/check-text")
async def check_text(file: UploadFile = File(None), text: str = Form(None), guideline_types: str = Form(None)):
    selected = []
    if guideline_types:
        try:
            user_selected = json.loads(guideline_types)
            if 'mutual_fund' in user_selected:
                selected.append(AMFI_CODE)
            if any(t in user_selected for t in ['investing', 'trading', 'ipo', 'fno_derivatives']):
                selected.append(STOCK_CODE)
            selected = list(set(selected))
        except Exception:
            pass
    selected.append(ASCI_CODE)  # Always include
    temp_path = None
    try:
        if file:
            ext = os.path.splitext(file.filename)[1] or ".txt"
            temp_path = f"/tmp/{uuid.uuid4()}{ext}"
            with open(temp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        elif text:
            temp_path = f"/tmp/{uuid.uuid4()}.txt"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            return aggregate_results([])
        results = check_compliance([temp_path], selected)
        return JSONResponse(content=results[0])
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# lightweight health endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "openai": bool(client)}

handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)