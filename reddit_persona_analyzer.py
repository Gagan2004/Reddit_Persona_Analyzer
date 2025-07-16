import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import praw
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# --- Data Structures ---
@dataclass
class Citation:
    type: str
    title: str
    content: str
    url: str
    subreddit: str
    score: int
    created_utc: float

@dataclass
class PersonaCharacteristic:
    value: Any
    confidence: float
    citations: List[Citation]

@dataclass
class UserPersona:
    username: str
    account_age_days: int
    total_karma: int
    age_range: PersonaCharacteristic
    gender: PersonaCharacteristic
    location: PersonaCharacteristic
    occupation: PersonaCharacteristic
    primary_interests: PersonaCharacteristic
    hobbies: PersonaCharacteristic
    favorite_subreddits: PersonaCharacteristic
    personality_traits: PersonaCharacteristic
    communication_style: PersonaCharacteristic
    posting_patterns: PersonaCharacteristic
    engagement_style: PersonaCharacteristic
    political_views: Optional[PersonaCharacteristic] = None
    goals_and_needs: Optional[PersonaCharacteristic] = None
    frustrations: Optional[PersonaCharacteristic] = None


class RedditPersonaAnalyzer:
    """
    Scrapes and analyzes a Reddit user's profile to generate a detailed persona.
    """
    def __init__(self, reddit_client_id: str, reddit_client_secret: str, reddit_user_agent: str, groq_api_key: Optional[str] = None):
        self.reddit = praw.Reddit(client_id=reddit_client_id, client_secret=reddit_client_secret, user_agent=reddit_user_agent)
        self.groq_api_key = groq_api_key
        self.groq_base_url = "https://api.groq.com/openai/v1/chat/completions"

    def extract_username_from_url(self, url: str) -> str:
        patterns = [r'reddit\.com/u(?:ser)?/([^/]+)', r'redd\.it/u(?:ser)?/([^/]+)', r'www\.reddit\.com/u(?:ser)?/([^/]+)']
        for pattern in patterns:
            if match := re.search(pattern, url):
                if (username := match.group(1)) and 3 <= len(username) <= 20:
                    return username
        raise ValueError(f"Could not extract a valid username from URL: {url}")

    def scrape_user_data(self, username: str, limit: int = 100) -> Optional[Dict]:
        try:
            user = self.reddit.redditor(username)
            user_info = {'username': user.name, 'account_created': user.created_utc, 'comment_karma': user.comment_karma, 'link_karma': user.link_karma, 'total_karma': user.comment_karma + user.link_karma}
            posts = [{'type': 'post', 'title': p.title, 'content': p.selftext, 'url': f"https://reddit.com{p.permalink}", 'subreddit': str(p.subreddit), 'score': p.score, 'created_utc': p.created_utc} for p in user.submissions.new(limit=limit)]
            comments = [{'type': 'comment', 'title': f"Comment in: {c.submission.title[:75]}...", 'content': c.body, 'url': f"https://reddit.com{c.permalink}", 'subreddit': str(c.subreddit), 'score': c.score, 'created_utc': c.created_utc} for c in user.comments.new(limit=limit)]
            return {'user_info': user_info, 'posts': posts, 'comments': comments}
        except Exception as e:
            logging.error(f"Failed to scrape data for user {username}: {e}")
            return None

    def _validate_and_repair_traits(self, traits: Any, original_content: List[Dict]) -> List[str]:
        """Validates AI trait analysis and triggers a repair if the response is generic."""
        if not isinstance(traits, list): return [str(traits)]
        
        generic_terms = {'high', 'medium', 'low', 'none', 'unknown', 'not specified', 'moderate'}
        if not traits or any(str(term).lower() in generic_terms for term in traits):
            logging.warning(f"Generic AI response detected: {traits}. Attempting self-correction...")
            repair_prompt = f"""The previous analysis returned the list: {traits}. This is not useful. Re-analyze the user content and provide a list of descriptive adjectives for their traits. DO NOT use generic levels like 'high', 'medium', or 'low'. INSTEAD, use words like 'analytical', 'creative', 'formal', 'concise'. User Content: {json.dumps(original_content[:5], indent=2)} Return a JSON object: {{"repaired_traits": ["descriptive", "adjectives"]}}"""
            try:
                payload = {"model": "llama3-70b-8192", "messages": [{"role": "system", "content": "You correct previous analysis to be more descriptive."}, {"role": "user", "content": repair_prompt}], "temperature": 0.3, "response_format": {"type": "json_object"}}
                response = requests.post(self.groq_base_url, headers={"Authorization": f"Bearer {self.groq_api_key}"}, json=payload, timeout=30)
                response.raise_for_status()
                new_traits = json.loads(response.json()['choices'][0]['message']['content']).get("repaired_traits", traits)
                logging.info(f"Self-correction successful. New traits: {new_traits}")
                return new_traits
            except Exception as e:
                logging.error(f"Self-correction API call failed: {e}")
                return traits
        return traits

    def analyze_with_groq(self, user_data: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Uses the Groq API for advanced analysis, requesting evidence indices."""
        if not self.groq_api_key: return {"error": "Groq API key not provided", "confidence": 0.0}
        sample_content = [{'subreddit': item['subreddit'], 'content': f"{item['title']} {item['content']}"[:500]} for item in (user_data['posts'] + user_data['comments'])[:20]]
        system_prompt = """You are a meticulous Reddit analyst. For each trait you identify, you MUST return the indices of the content samples that support your conclusion. Your response must be a single, valid JSON object. Example for 'personality': {"personality_traits": ["analytical", "inquisitive"], "traits_evidence_indices": [2, 5], "communication_style": ["formal", "detailed"], "style_evidence_indices": [1, 3], "confidence": 0.85}. For traits, use descriptive adjectives. DO NOT use generic levels like 'high', 'medium', or 'low'. If no evidence is found, use "unknown" or an empty list."""
        prompts = {'demographics': "Analyze for an estimated age range, gender, location, and occupation based on contextual clues. For age, infer a range (e.g., 'Late Teens', '20s', '30-40') from interests, cultural references, and writing style. For all traits, provide `_evidence_indices` pointing to the content that provides these clues.", 'interests': "Analyze for primary interests and hobbies.", 'personality': "Analyze for personality traits (e.g. 'humorous', 'analytical') and communication style (e.g. 'formal', 'concise').", 'goals_frustrations': "Analyze for goals, needs, and frustrations."}
        try:
            payload = {"model": "llama3-70b-8192", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Analyze for {prompts[category]}. Provide `_evidence_indices` for each trait.\n\nUser Content Samples (with index):\n{json.dumps(list(enumerate(sample_content)), indent=2)}"}], "temperature": 0.2, "response_format": {"type": "json_object"}}
            response = requests.post(self.groq_base_url, headers={"Authorization": f"Bearer {self.groq_api_key}"}, json=payload, timeout=45)
            response.raise_for_status()
            return json.loads(response.json()['choices'][0]['message']['content'])
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
            logging.error(f"Groq API call failed for category '{category}': {e}")
            return {"error": f"API interaction failed: {e}", "confidence": 0.1}

    def build_persona(self, user_data: Dict) -> UserPersona:
        """Builds the UserPersona by orchestrating all analyses."""
        if not user_data or (not user_data.get('posts') and not user_data.get('comments')): raise ValueError("No posts or comments found.")
        user_info, posts, comments = user_data['user_info'], user_data['posts'], user_data['comments']
        all_content = posts + comments

        logging.info("Analyzing with Groq AI...")
        groq_demographics = self.analyze_with_groq(user_data, 'demographics')
        groq_interests = self.analyze_with_groq(user_data, 'interests')
        groq_personality = self.analyze_with_groq(user_data, 'personality')
        groq_goals = self.analyze_with_groq(user_data, 'goals_frustrations')
        
        repaired_traits = self._validate_and_repair_traits(groq_personality.get('personality_traits', []), all_content)
        repaired_style = self._validate_and_repair_traits(groq_personality.get('communication_style', []), all_content)
        
        def get_citations(trait_name: str, ai_response: Dict, fallback_keywords: List[str]) -> List[Citation]:
            """Hybrid strategy to get the best citations for a trait."""
            citations = []
            if indices := ai_response.get(f"{trait_name}_evidence_indices", []):
                for idx in indices:
                    if isinstance(idx, int) and 0 <= idx < len(all_content): citations.append(self._create_citation(all_content[idx]))
                if citations: return citations
            if citations := self._extract_evidence_by_keyword(fallback_keywords, all_content): return citations
            return self._get_general_evidence(all_content) if not fallback_keywords else []

        persona = UserPersona(
            username=user_info['username'], account_age_days=int((time.time() - user_info['account_created']) / 86400), total_karma=user_info['total_karma'],
            age_range=self._build_characteristic(groq_demographics.get('age_range', 'unknown'), groq_demographics.get('confidence', 0.3), get_citations('age', groq_demographics, ['year old', 'my age', 'born in'])),
            gender=self._build_characteristic(groq_demographics.get('gender', 'unknown'), groq_demographics.get('confidence', 0.3), get_citations('gender', groq_demographics, ['i am a', 'as a man', 'as a woman'])),
            location=self._build_characteristic(groq_demographics.get('location', 'unknown'), groq_demographics.get('confidence', 0.3), get_citations('location', groq_demographics, ['live in', 'from', 'my city'])),
            occupation=self._build_characteristic(groq_demographics.get('occupation', 'unknown'), groq_demographics.get('confidence', 0.3), get_citations('occupation', groq_demographics, ['my job', 'i work as', 'career'])),
            primary_interests=self._build_characteristic(groq_interests.get('primary_interests', []), groq_interests.get('confidence', 0.5), get_citations('interests', groq_interests, [])),
            hobbies=self._build_characteristic(groq_interests.get('hobbies', []), groq_interests.get('confidence', 0.5), get_citations('hobbies', groq_interests, ['my hobby', 'i enjoy'])),
            personality_traits=self._build_characteristic(repaired_traits, groq_personality.get('confidence', 0.5), get_citations('traits', groq_personality, [])),
            communication_style=self._build_characteristic(repaired_style, groq_personality.get('confidence', 0.5), get_citations('style', groq_personality, [])),
            favorite_subreddits=self._build_characteristic(list(self._analyze_subreddit_activity(all_content).keys()), 0.95, self._get_general_evidence(all_content)),
            posting_patterns=self._build_characteristic(self._analyze_posting_patterns(all_content), 0.95, []),
            engagement_style=self._build_characteristic(self._analyze_engagement_style(posts, comments), 0.8, []),
        )
        if groq_goals.get('confidence', 0) > 0.3:
            persona.goals_and_needs = self._build_characteristic(groq_goals.get('goals_and_needs', []), groq_goals.get('confidence', 0.4), get_citations('goals', groq_goals, ['i want', 'i need', 'hope to']))
            persona.frustrations = self._build_characteristic(groq_goals.get('frustrations', []), groq_goals.get('confidence', 0.4), get_citations('frustrations', groq_goals, ['i hate', 'annoying', 'frustrating']))
        return persona

    def _generate_summary(self, persona: UserPersona) -> str:
        """Creates a one-line summary of the persona."""
        parts = []
        if (gender := self._normalize_value(persona.gender.value)) not in ["unknown", "Not specified"]: parts.append(gender.capitalize())
        if (age := self._normalize_value(persona.age_range.value)) not in ["unknown", "Not specified"]: parts.append(f"in their {age}")
        if (location := self._normalize_value(persona.location.value)) not in ["unknown", "Not specified"]: parts.append(f"from {location}")
        summary = ' '.join(parts)
        if interests := self._normalize_value(persona.primary_interests.value):
            if interests != "Not specified": summary += f", interested in {interests}."
        return summary if summary else "No high-level summary could be generated."

    def save_persona_to_txt(self, persona: UserPersona, filename: str):
        """Saves the generated persona to a polished, human-readable text file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n📊 REDDIT USER PERSONA: /u/{persona.username}\n{'='*70}\n\n")
            f.write(f"SUMMARY: {self._generate_summary(persona)}\n\n")
            f.write(f"📅 Account Age: {persona.account_age_days} days\n⭐ Total Karma: {persona.total_karma:,}\n\n")
            all_characteristics = [
                ("👤 Demographics", [("Age Range", persona.age_range), ("Gender", persona.gender), ("Location", persona.location), ("Occupation", persona.occupation)]),
                ("🎯 Interests & Hobbies", [("Primary Interests", persona.primary_interests), ("Hobbies", persona.hobbies), ("Favorite Subreddits", persona.favorite_subreddits)]),
                ("🧠 Personality & Behavior", [("Inferred Traits", persona.personality_traits), ("Communication Style", persona.communication_style)]),
                ("🌟 Goals & Frustrations", [("Goals & Needs", persona.goals_and_needs), ("Frustrations", persona.frustrations)]),
                ("📈 Content & Activity Analysis", [("Posting Patterns", persona.posting_patterns), ("Engagement Style", persona.engagement_style)])
            ]
            max_len = 0
            for _, char_list in all_characteristics:
                for name, char in char_list:
                    if char and self._normalize_value(char.value) != "Not specified":
                        if len(name) > max_len: max_len = len(name)
            for title, char_list in all_characteristics:
                self._write_section(f, title, char_list, max_len)
        logging.info(f"SUCCESS: Text persona for '/u/{persona.username}' saved to '{filename}'.")
    
    def save_persona_to_json(self, persona: UserPersona, filename: str):
        """Saves the complete persona object to a structured JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            persona_dict = asdict(persona)
            json.dump(persona_dict, f, indent=4)
        logging.info(f"SUCCESS: JSON persona for '/u/{persona.username}' saved to '{filename}'.")

    def _write_section(self, f, title: str, characteristics: List[Tuple[str, Optional[PersonaCharacteristic]]], padding: int):
        f.write(f"--- {title} ---\n")
        printable_chars = [(name, char) for name, char in characteristics if char and self._normalize_value(char.value) != "Not specified"]
        if not printable_chars:
            f.write("  No specific data found for this category.\n\n")
            return
        for name, char in printable_chars:
            self._write_characteristic(f, name, char, padding)

    def _write_characteristic(self, f, name: str, characteristic: PersonaCharacteristic, padding: int):
        """Writes a single, fully formatted characteristic block to the file."""
        is_general_evidence = False
        if characteristic.citations and self._normalize_value(characteristic.value) not in ["unknown", "Not specified"]:
            keywords = {'Age Range': ['age', 'born'], 'Gender': ['man', 'woman', 'guy', 'girl'], 'Location': ['live', 'from', 'city', 'country'], 'Occupation':['job', 'work', 'career']}.get(name, [])
            if keywords and not any(kw in (c.title + c.content).lower() for c in characteristic.citations for kw in keywords):
                is_general_evidence = True
        
        f.write(f"  - {name:<{padding}} : {self._normalize_value(characteristic.value)} (Confidence: {characteristic.confidence:.0%})\n")
        
        if characteristic.citations:
            evidence_label = "General Evidence (Top Posts)" if is_general_evidence else "Evidence"
            f.write(f"    {evidence_label}:\n")
            for i, cite in enumerate(characteristic.citations[:2], 1):
                f.write(f"      {i}. [{cite.subreddit}] {cite.title}\n         URL: {cite.url}\n")
        f.write("\n")

    # --- Helper & Citation & Analysis Methods ---
    def _create_citation(self, item: Dict) -> Citation: return Citation(type=item['type'], title=item['title'], content=item['content'][:500], url=item['url'], subreddit=item['subreddit'], score=item['score'], created_utc=item['created_utc'])
    def _build_characteristic(self, value: Any, confidence: float, citations: List[Citation]) -> PersonaCharacteristic: return PersonaCharacteristic(value=value, confidence=confidence, citations=citations)
    def _normalize_value(self, value: Any) -> str:
        if isinstance(value, list): return ', '.join([str(v).strip() for v in value if str(v).strip()]) or "Not specified"
        return str(value) if value else "Not specified"
    def _get_general_evidence(self, all_content: List[Dict]) -> List[Citation]: return [self._create_citation(item) for item in sorted(all_content, key=lambda x: x.get('score', 0), reverse=True)[:3]]
    def _extract_evidence_by_keyword(self, keywords: List[str], all_content: List[Dict]) -> List[Citation]:
        if not keywords: return []
        evidence = []
        for item in all_content:
            if any(kw in (item.get('title', '') + ' ' + item.get('content', '')).lower() for kw in keywords): evidence.append(self._create_citation(item))
        return evidence[:3]
    def _analyze_subreddit_activity(self, all_content: List[Dict]) -> Dict: return {'most_active_subreddits': dict(Counter(item['subreddit'] for item in all_content).most_common(5))}
    def _analyze_posting_patterns(self, all_content: List[Dict]) -> str:
        if not all_content: return "No recent activity"
        peak_hour, _ = Counter(datetime.fromtimestamp(item['created_utc']).hour for item in all_content).most_common(1)[0]
        return f"Most active around {peak_hour:02d}:00 UTC"
    def _analyze_engagement_style(self, posts: List[Dict], comments: List[Dict]) -> str:
        if not posts and not comments: return "No recent activity"
        avg_post_score = sum(p['score'] for p in posts) / len(posts) if posts else 0
        avg_comment_score = sum(c['score'] for c in comments) / len(comments) if comments else 0
        if avg_post_score > 50 or avg_comment_score > 10: return "Content is generally well-received"
        return "Receives moderate engagement"

def main():
    """Main function to configure logging, parse arguments, and run the analyzer."""
    # --- Configure Logging ---
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('scraper.log')
    file_handler.setFormatter(log_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    # Prevent adding handlers multiple times if this function is called again
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Reddit User Persona Analyzer (Final Version)")
    parser.add_argument('profile_url', help="Full URL of the Reddit user profile to analyze.")
    parser.add_argument('--limit', '-l', type=int, default=100, help="Max number of posts/comments to fetch.")
    parser.add_argument('--format', choices=['txt', 'json'], default='txt', help="Output format: 'txt' for human-readable report or 'json' for structured data.")
    args = parser.parse_args()

    # --- Main Execution Block ---
    try:
        analyzer = RedditPersonaAnalyzer(os.getenv('REDDIT_CLIENT_ID'), os.getenv('REDDIT_CLIENT_SECRET'), os.getenv('REDDIT_USER_AGENT', 'PersonaAnalyzer/Final'), os.getenv('GROQ_API_KEY'))
        
        username = analyzer.extract_username_from_url(args.profile_url)
        logging.info(f"Starting analysis for /u/{username}")

        output_filename = f"persona_{username}.{args.format}"
        
        if not (user_data := analyzer.scrape_user_data(username, limit=args.limit)): sys.exit(1)
        
        if not user_data['posts'] and not user_data['comments']:
            logging.warning(f"No recent activity found for user. Cannot generate persona."); sys.exit(0)
            
        persona = analyzer.build_persona(user_data)
        
        if args.format == 'json':
            analyzer.save_persona_to_json(persona, output_filename)
        else:
            analyzer.save_persona_to_txt(persona, output_filename)
        
    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}", exc_info=True); sys.exit(1)

if __name__ == "__main__":
    main()