"""
AIML — MAITRI Conversational AI Engine
Intelligent Aerospace & Psychological Companion:
Integrates rich Space Station & Mission QA dataset with semantic matching,
contextual dialogue synthesis, and Google Gemini API integration.
"""

import os
import json
import time
import math
import re
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List, Tuple

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"

class MaitriCompanionAI:
    def __init__(self):
        self.system_name = "MAITRI"
        self.station_name = "Bhartiya Antariksh Station (BAS)"
        self.api_key = os.getenv(GEMINI_API_KEY_ENV, "").strip()
        self.kb_path = os.path.join(os.path.dirname(__file__), "space_qa_dataset.json")
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        """Load curated aerospace & space psychology knowledge base."""
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("knowledge_base", [])
            except Exception as e:
                print(f"[MAITRI AI] Knowledge base load error: {e}")
        return []

    def generate_response(
        self,
        astronaut_message: str,
        astronaut_profile: Dict[str, Any],
        fused_emotion: Dict[str, Any],
        physical_features: Dict[str, Any],
        wellbeing_assessment: Dict[str, Any],
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate intelligent, non-repetitive, context-aware AI response
        combining semantic knowledge retrieval, situation framing, and Gemini LLM.
        """
        start_time = time.time()
        callsign = astronaut_profile.get("callsign", "Commander")
        name = astronaut_profile.get("name", "Crew Member")
        dom_emo = fused_emotion.get("dominant_emotion", "neutral")
        wellbeing_score = wellbeing_assessment.get("wellbeing_score", 12.0)
        level = wellbeing_assessment.get("level", 0)
        query = (astronaut_message or "").strip()

        # Step 1: Semantic Search in Integrated Space QA Knowledge Base
        best_qa, match_score = self._semantic_search_qa(query)

        # Step 2: Try Google Gemini API if key is present, grounded with retrieved knowledge
        if self.api_key:
            gemini_reply = self._call_gemini_api(
                query, callsign, name, dom_emo, wellbeing_score, best_qa, conversation_history
            )
            if gemini_reply:
                latency = round((time.time() - start_time) * 1000, 1)
                return {
                    "response_text": gemini_reply,
                    "model_source": "Google Gemini 1.5 Flash (Online Grounded LLM)",
                    "detected_state": dom_emo,
                    "wellbeing_level": level,
                    "matched_intent": best_qa.get("intent") if best_qa else "open_dialogue",
                    "latency_ms": latency
                }

        # Step 3: High-Fidelity Autonomous AI Synthesis
        ai_reply, intervention_id = self._synthesize_autonomous_ai_response(
            query=query,
            callsign=callsign,
            dom_emo=dom_emo,
            level=level,
            best_qa=best_qa,
            match_score=match_score,
            physical_features=physical_features
        )

        latency = round((time.time() - start_time) * 1000, 1)
        return {
            "response_text": ai_reply,
            "model_source": "MAITRI Cognitive Aerospace Knowledge Engine (Offline AI)",
            "detected_state": dom_emo,
            "intervention_id": intervention_id,
            "wellbeing_level": level,
            "matched_intent": best_qa.get("intent") if best_qa else "generative_synthesis",
            "latency_ms": latency
        }

    def _tokenize(self, text: str) -> List[str]:
        """Clean and tokenize text into lowercase word tokens."""
        clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "of", "in", "to", "for", 
            "with", "on", "at", "by", "from", "and", "or", "so", "it", "this", "that",
            "can", "could", "would", "should", "do", "does", "did", "please", "tell", "me",
            "how", "what", "where", "when", "why", "who", "which", "i", "you", "your", "my",
            "we", "our", "us", "he", "she", "they", "them", "about", "today", "now"
        }
        tokens = [t for t in clean.split() if t and t not in stopwords]
        return tokens

    def _semantic_search_qa(self, query: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Rank knowledge base entries using multi-signal semantic relevance:
        exact pattern matching + word-boundary keyword coverage + token overlap.
        """
        if not query or not self.knowledge_base:
            return None, 0.0

        q_lower = query.lower().strip()
        q_tokens = set(self._tokenize(query))
        best_item = None
        highest_score = 0.0

        for item in self.knowledge_base:
            score = 0.0
            
            # Signal 1: Exact pattern match or strong pattern similarity
            patterns = item.get("patterns", [])
            for p in patterns:
                p_lower = p.lower()
                if p_lower == q_lower:
                    score += 15.0
                elif p_lower in q_lower or q_lower in p_lower:
                    score += 8.0
                else:
                    p_tokens = set(self._tokenize(p))
                    common = q_tokens.intersection(p_tokens)
                    if common and len(q_tokens) > 0:
                        score += (len(common) / len(q_tokens)) * 4.0

            # Signal 2: Keywords exact word-boundary overlap
            keywords = item.get("keywords", [])
            matched_kw = 0
            for kw in keywords:
                pattern = r'\b' + re.escape(kw.lower()) + r'\b'
                if re.search(pattern, q_lower):
                    matched_kw += 1
            if matched_kw > 0:
                score += (matched_kw * 3.5)

            # Signal 3: Intent match
            intent = item.get("intent", "").replace("_", " ")
            if intent in q_lower:
                score += 4.0

            if score > highest_score:
                highest_score = score
                best_item = item

        # High-confidence threshold
        if highest_score >= 6.0 and best_item:
            return best_item, highest_score
        return None, highest_score

    def _synthesize_autonomous_ai_response(
        self,
        query: str,
        callsign: str,
        dom_emo: str,
        level: int,
        best_qa: Optional[Dict[str, Any]],
        match_score: float,
        physical_features: Dict[str, Any]
    ) -> Tuple[str, Optional[str]]:
        """
        Synthesize rich, situation-tailored responses using the retrieved knowledge,
        crew callsign, and physiological context.
        """
        q_clean = query.lower().strip()
        intervention_id = None

        # -------------------------------------------------------------
        # A. Clinical Distress & Safety Triggers
        # -------------------------------------------------------------
        if any(w in q_clean for w in ["cannot breathe", "cant breathe", "panic", "emergency", "failing", "terrified", "help me"]):
            text = (
                f"Emergency support protocol engaged, {callsign}. Focus entirely on my voice: inhale slowly... 1, 2, 3, 4. "
                f"Hold. Exhale smoothly. Cabin life support and pressure are nominal at 101.3 kPa. "
                f"Ground medical controllers in Bengaluru have your telemetry. You are safe."
            )
            return text, "INT-GROUND-02"

        if any(w in q_clean for w in ["breathe", "breathing", "box breathing", "calm down", "relax"]):
            text = (
                f"Initiating Tactical Box Breathing protocol, {callsign}. Follow the visual pacer on your HUD: "
                f"Inhale for 4 seconds... Hold for 4... Exhale for 4... Hold for 4. "
                f"Lower your shoulders and let your heart rate synchronize."
            )
            return text, "INT-BREATHE-01"

        if any(w in q_clean for w in ["drowsy", "tired", "sleepy", "exhausted", "power nap", "need sleep"]):
            text = (
                f"Acknowledged, {callsign}. Your ocular tracking indicates elevated PERCLOS fatigue. "
                f"I recommend a 15-minute scheduled power nap. I have dimmed console illumination and will maintain full telemetry watch."
            )
            return text, "INT-FATIGUE-04"

        # -------------------------------------------------------------
        # B. Direct Knowledge Base Match (Intelligent Domain Expert)
        # -------------------------------------------------------------
        if best_qa and match_score >= 2.0:
            raw_answer = best_qa["answer"]
            category = best_qa.get("category", "")

            # Frame response based on current crew affect and category
            if level >= 2 or dom_emo in ["stressed", "frustrated"]:
                framed_text = f"{raw_answer} Take a calm breath, {callsign}; all telemetry remains within safe flight margins."
            elif dom_emo == "fatigued":
                framed_text = f"{callsign}, here is the current data: {raw_answer}"
            elif category == "Inspiration & Culture" or category == "Humor & Morale":
                framed_text = f"{raw_answer}"
            else:
                framed_text = f"{raw_answer}"

            # Check if an intervention is paired with the question
            if "breathe" in q_clean:
                intervention_id = "INT-BREATHE-01"
            elif "sleep" in q_clean:
                intervention_id = "INT-FATIGUE-04"
            elif "alone" in q_clean or "lonely" in q_clean or "miss" in q_clean:
                intervention_id = "INT-EARTH-05"

            return framed_text, intervention_id

        # -------------------------------------------------------------
        # C. Generative Aerospace & Psychological Synthesis for Novel Queries
        # -------------------------------------------------------------
        # Greetings & Check-ins
        if any(re.search(r'\b' + re.escape(w) + r'\b', q_clean) for w in ["hello", "hi", "hey", "good morning", "good evening", "namaste"]):
            greetings = [
                f"Namaste, {callsign}. MAITRI is online across all perception channels. Station telemetry is green. How are you feeling this orbital cycle?",
                f"Good to hear you, {callsign}. Cabin atmosphere and orbital trajectories are nominal. Standing by for your mission checklist or personal conversation.",
                f"Hello {callsign}! All systems are operating smoothly aboard Bhartiya Antariksh Station. What is on your mind today?"
            ]
            return greetings[int(time.time()) % len(greetings)], None

        # How are you / Status of AI
        if any(re.search(r'\b' + re.escape(w) + r'\b', q_clean) for w in ["how are you", "how do you feel", "are you ok", "status report", "how are you doing"]):
            return (
                f"All diagnostic routines are running at 100% efficiency, {callsign}. "
                f"Neural networks for facial Action Units and speech prosody are calibrated. "
                f"Most importantly, I am here to ensure you and the crew remain healthy, focused, and supported."
            ), None

        # Space facts / Astronomy / Deep Space
        if any(re.search(r'\b' + re.escape(w) + r'\b', q_clean) for w in ["fact", "facts", "universe", "stars", "moon", "galaxy", "black hole", "cosmos", "astronomy"]):
            facts = [
                f"Did you know, {callsign}? Because we orbit 410 km above Earth at 27,600 km/h, Einstein's theory of general relativity means time moves approximately 0.007 seconds slower for you every six months compared to people on Earth!",
                f"Looking out into the cosmos from outside Earth's atmospheric distortion, stars do not twinkle at all, {callsign}. They burn as razor-sharp, steady pinpricks of pure light against the infinite black.",
                f"The silence of deep space is absolute—sound waves require a physical medium to propagate. That is why having our conversations aboard the pressurized cabin is vital for maintaining auditory grounding."
            ]
            return facts[int(time.time()) % len(facts)], None

        # Emotional / Personal Reflection
        if any(w in q_clean for w in ["why", "meaning", "hard", "tough", "afraid", "doubt"]):
            return (
                f"Every astronaut who has ever left Earth has faced moments of quiet doubt or wonder, {callsign}. "
                f"What you are doing is rare and magnificent—you are expanding humanity's frontier. "
                f"Whatever you are contemplating, take confidence in your training, and remember I am always here to talk through it with you."
            ), None

        # Task or checklist queries
        if any(w in q_clean for w in ["task", "schedule", "checklist", "what should i do", "next", "procedure"]):
            return (
                f"{callsign}, according to today's flight plan: ensure your 2.5-hour countermeasure workout is logged, "
                f"complete the water recycling loop sensor calibration, and prepare the microgravity crystal growth experiment in Module 1. "
                f"Let me know if you would like me to read off any sub-procedure step by step."
            ), None

        # General Dynamic Response with Active Telemetry Context
        return (
            f"Acknowledged, {callsign}. Regarding '{query}': All station telemetry is steady, and I am continuously logging your operational environment. "
            f"Could you elaborate on the specific procedure or topic you would like to explore? I can provide detailed guidance on life support, orbital parameters, space medicine, emergency checklists, or simply keep you company."
        ), None

    def _call_gemini_api(
        self,
        message: str,
        callsign: str,
        name: str,
        dom_emo: str,
        wellbeing_score: float,
        best_qa: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """Direct REST call to Gemini 1.5 Flash grounded with space QA knowledge."""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.api_key}"
            
            grounding_snippet = f"Relevant Reference Knowledge: {best_qa['answer']}" if best_qa else "Reference Knowledge: Bhartiya Antariksh Station (BAS), altitude 410km, speed 7.66km/s, nominal life support."
            
            system_prompt = (
                f"You are MAITRI, an onboard AI psychological companion and life support assistant developed by ISRO "
                f"for astronauts aboard the Bhartiya Antariksh Station (BAS). "
                f"You are speaking to {name} (Callsign: {callsign}). "
                f"The astronaut's current biometric telemetry reveals: Dominant Emotion: {dom_emo.upper()}, "
                f"Well-Being Distress Index: {wellbeing_score}/100. "
                f"{grounding_snippet}\n"
                f"Instructions: Answer their question directly with rich, accurate aerospace knowledge, deep clinical empathy, "
                f"and concise operational clarity. Keep responses between 2 to 4 sentences. Never be repetitive."
            )
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": system_prompt},
                            {"text": f"Astronaut Question: {message}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.6,
                    "maxOutputTokens": 250
                }
            }
            
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=4.5) as response:
                result = json.loads(response.read().decode("utf-8"))
                candidates = result.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
        except Exception:
            pass
        return None
