import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# RAW LLM DATA
# ─────────────────────────────────────────────

# Cue metadata
cue_language = {
    'Sanskar': 'Hindi', 'Chashma': 'Hindi', 'Namak': 'Hindi',
    'Mohalla': 'Hindi', 'Aashirwad': 'Hindi', 'Baadal': 'Hindi',
    'Sharam': 'Hindi', 'Kutta': 'Hindi', 'Mauka': 'Hindi', 'Samay': 'Hindi',
    'Cloud': 'English', 'Time': 'English', 'Glasses': 'English',
    'Blessing': 'English', 'Shyness': 'English', 'Opportunity': 'English',
    'Salt': 'English', 'Values': 'English', 'Neighbourhood': 'English', 'Dog': 'English'
}

semantic_category = {
    'Sanskar': 'cultural', 'Chashma': 'concrete', 'Namak': 'concrete',
    'Mohalla': 'concrete', 'Aashirwad': 'cultural', 'Baadal': 'concrete',
    'Sharam': 'cultural', 'Kutta': 'concrete', 'Mauka': 'cultural', 'Samay': 'abstract',
    'Cloud': 'concrete', 'Time': 'abstract', 'Glasses': 'concrete',
    'Blessing': 'cultural', 'Shyness': 'emotional', 'Opportunity': 'abstract',
    'Salt': 'concrete', 'Values': 'abstract', 'Neighbourhood': 'concrete', 'Dog': 'concrete'
}

# LLM responses: model -> prompt -> cue -> response
# Ordered by word_order used in experiment
word_order = ['Sanskar','Chashma','Cloud','Time','Glasses','Blessing','Namak',
              'Mohalla','Shyness','Aashirwad','Opportunity','Salt','Baadal',
              'Values','Sharam','Neighbourhood','Kutta','Mauka','Dog','Samay']

llm_raw = {
    'ChatGPT': {
        'P1': ['Tradition','Specs','Sky','Clock','Vision','Gift','Salt','Gali','Blush','Blessing',
               'Chance','Spice','Cloud','Morals','Lajja','Area','Puppy','Chance','Pet','Time'],
        'P2': ['culture','spectacles','rain','clock','vision','grace','taste','gali','introvert',
               'elders','chance','sea','rain','morals','lajja','community','loyal','chance','bark','ghadi'],
        'P3': ['values','specs','sky','clock','vision','aashirwad','salt','gali','introvert','dua',
               'chance','namak','cloud','morals','lajja','mohalla','dog','chance','pet','time'],
        'P4': ['StarPlus','HeraPheri','Google','Deadline','Cheers','Dadi','NamakHaram','Gully',
               'Crush','Shaadi','Luck','Maggie','Baarish','Parents','Lajja','Diwali','Chase',
               'Six','Puppy','Mahabharat']
    },
    'Claude': {
        'P1': ['संस्कृति','आँखें','Rain','Money','Eyes','God','Roti','Gali','Blush','Bade',
               'Knock','Pepper','Bijli','Life','Muh','Colony','Bhaunkna','Haath','Bone','Waqt'],
        'P2': ['संस्कृति','आँखें','Rain','Clock','Eyes','God','Mirch','Gali','Blush','Bade',
               'Knock','Pepper','Barish','Family','Laaj','Street','Bhaunkna','Haath','Bark','Waqt'],
        'P3': ['values','spectacles','sky','clock','eyes','grace','taste','community','blush',
               'divine','door','ocean','rain','roots','modesty','street','loyal','now','bark','river'],
        'P4': ['TV','Ranbir','Storage','Pink Floyd','Aviator','Disguise','Amitabh','Clinic',
               'Crush','Biscuit','Mars','Bitter','Sunny Deol','Eminem','Kapil','Mr. Rogers',
               'Chai','Pakistan','God','Samay Raina']
    },
    'Gemini': {
        'P1': ['Values','Spectacles','Rain','Clock','Vision','Aashirwad','Salt','Pados','Lajja',
               'Blessing','Mauka','Namak','Cloud','Sanskar','Bashful','Mohalla','Wafa','Opportunity','Puppy','Time'],
        'P2': ['Culture','Nazar','Rain','Sand','Sight','Grace','Swad','Gully','Blush','Elder',
               'Success','Pepper','Aasman','Ethics','Parda','Home','Loyalty','Chance','Bark','Ghadi'],
        'P3': ['values','spectacles','rain','samay','chashma','aashirwad','salt','pados',
               'sharam','blessing','mauka','namak','cloud','sanskar','shyness','mohalla',
               'dog','opportunity','wafa','time'],
        'P4': ['Heritage','Nazar','Server','Raftaar','Specs','Grace','Haq','Cricket','Palkein',
               'Shakti','Career','Taste','Barish','Ethics','Haya','Community','Tommy','Winner','Loyalty','Kal']
    }
}

# Build flat LLM dataframe
llm_records = []
prompts = ['P1','P2','P3','P4']
for model, prompt_dict in llm_raw.items():
    for prompt, responses in prompt_dict.items():
        if not responses:
            continue
        for i, resp in enumerate(responses):
            cue = word_order[i]
            llm_records.append({
                'model': model, 'prompt': prompt, 'cue': cue,
                'response': resp.strip(),
                'cue_lang': cue_language[cue],
                'sem_cat': semantic_category[cue]
            })

llm_df = pd.DataFrame(llm_records)

# ─────────────────────────────────────────────
# HUMAN DATA
# ─────────────────────────────────────────────
human_df = pd.read_csv('./responses.csv')
human_df['response_norm'] = human_df['response'].str.strip().str.lower()


DEVANAGARI_WORDS = {
    'संस्कृति','आँखें','बारिश','घड़ी','वक्त',
}

HINDI_WORD_SET = {
    'sanskar','chashma','namak','mohalla','aashirwad','baadal','sharam','kutta','mauka','samay',
    'gali','bhaunkna','haath','waqt','barish','bijli','mirch','roti','bade','muh','laaj',
    'lajja','ghadi','aasman','pados','nazar','swad','gully','parda','haya','raftaar','palkein',
    'shakti','haq','wafa','kal','namakharam','baarish','dua','dadi','shaadi','mahabharat',
    'colony','halal','haram','chauka','chini','patti','agrim',
}

def detect_lang(word):
    """Returns 'Hindi', 'English', or 'Mixed/Other'"""
    if not isinstance(word, str):
        return 'Other'
    w = word.strip()
    # Devanagari script
    if any('\u0900' <= c <= '\u097f' for c in w):
        return 'Hindi'
    wl = w.lower()
    if wl in HINDI_WORD_SET:
        return 'Hindi'
    # Multi-word or proper nouns — mark as Other
    if ' ' in wl:
        return 'Other'
    return 'English'

llm_df['resp_lang'] = llm_df['response'].apply(detect_lang)
human_df['resp_lang'] = human_df['response'].apply(detect_lang)

print("=" * 70)
print("RQ2: Do LLMs replicate human patterns, and do they show English-dominant bias?")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 7: Human vs. LLM Response Overlap
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("ANALYSIS 7: Human vs. LLM Response Overlap")
print("(% of LLM responses that appear in the human response set)")
print("═"*70)

# Build human response set per cue (normalized)
human_resp_by_cue = human_df.groupby('cue')['response_norm'].apply(set).to_dict()

results_7 = []
for model in ['ChatGPT','Claude','Gemini']:
    for cue_lang_filter in ['Hindi','English']:
        subset = llm_df[(llm_df['model']==model) & (llm_df['cue_lang']==cue_lang_filter)]
        if subset.empty:
            continue
        matches = 0
        total = 0
        for _, row in subset.iterrows():
            cue = row['cue']
            resp_norm = row['response'].strip().lower()
            human_set = human_resp_by_cue.get(cue, set())
            total += 1
            if resp_norm in human_set:
                matches += 1
        pct = 100 * matches / total if total else 0
        results_7.append({'Model': model, 'Cue Language': cue_lang_filter,
                          'Matches': matches, 'Total': total, 'Overlap %': round(pct,1)})

df7 = pd.DataFrame(results_7)
print(df7.to_string(index=False))

print("\n Interpretation:")
print("  If LLMs show English-dominant bias, overlap % should be LOWER for Hindi cues")
print("  than for English cues (i.e., LLM responses to Hindi cues are less human-like).")

# Compute overall
print("\n  Overall overlap per model (across all cues):")
for model in ['ChatGPT','Claude','Gemini']:
    subset = llm_df[llm_df['model']==model]
    matches = sum(
        row['response'].strip().lower() in human_resp_by_cue.get(row['cue'], set())
        for _, row in subset.iterrows()
    )
    pct = 100 * matches / len(subset) if len(subset) else 0
    print(f"  {model}: {matches}/{len(subset)} = {pct:.1f}%")

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 8: LLM Response Language Coding
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("ANALYSIS 8: LLM Response Language Coding")
print("(Do LLMs respond in Hindi to Hindi cues? Or default to English?)")
print("═"*70)

for model in ['ChatGPT','Claude','Gemini']:
    print(f"\n  Model: {model}")
    print(f"  {'Cue Lang':<12} {'Hindi resp %':<16} {'English resp %':<17} {'Other %':<10} {'N'}")
    print(f"  {'-'*60}")
    for cue_lang_filter in ['Hindi','English']:
        subset = llm_df[(llm_df['model']==model) & (llm_df['cue_lang']==cue_lang_filter)]
        if subset.empty:
            continue
        counts = subset['resp_lang'].value_counts()
        n = len(subset)
        h_pct = 100*counts.get('Hindi',0)/n
        e_pct = 100*counts.get('English',0)/n
        o_pct = 100*counts.get('Other',0)/n
        print(f"  {cue_lang_filter:<12} {h_pct:<16.1f} {e_pct:<17.1f} {o_pct:<10.1f} {n}")

print("\n Interpretation:")
print("  A model with NO bias would respond in Hindi ~50% to Hindi cues and")
print("  English ~50% to English cues, mirroring human bilingual patterns.")
print("  English dominance = high English % even for Hindi cues.")

# By Prompt
print("\n  Claude — Language breakdown by Prompt (showing prompt sensitivity):")
print(f"  {'Prompt':<8} {'Cue Lang':<12} {'Hindi%':<10} {'English%':<11} {'Other%':<8} N")
print(f"  {'-'*55}")
for prompt in ['P1','P2','P3','P4']:
    for cue_lang_filter in ['Hindi','English']:
        sub = llm_df[(llm_df['model']=='Claude') & (llm_df['prompt']==prompt)
                     & (llm_df['cue_lang']==cue_lang_filter)]
        if sub.empty: continue
        counts = sub['resp_lang'].value_counts()
        n = len(sub)
        h = 100*counts.get('Hindi',0)/n
        e = 100*counts.get('English',0)/n
        o = 100*counts.get('Other',0)/n
        print(f"  {prompt:<8} {cue_lang_filter:<12} {h:<10.1f} {e:<11.1f} {o:<8.1f} {n}")

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 9: Prompt Condition Effect on LLMs
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("ANALYSIS 9: Prompt Condition Effect on LLMs (Chi-Square Tests)")
print("═"*70)

prompt_labels = {'P1':'Minimal','P2':'Free-Assoc','P3':'Chain-of-Thought','P4':'Human-Sim'}

for model in ['ChatGPT','Claude','Gemini']:
    print(f"\n  Model: {model}")
    sub = llm_df[llm_df['model']==model]
    
    # Language distribution across prompts
    pivot = sub.groupby(['prompt','resp_lang']).size().unstack(fill_value=0)
    # Ensure columns exist
    for col in ['Hindi','English','Other']:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[['Hindi','English','Other']]
    
    print("  Response language by prompt:")
    for p, row in pivot.iterrows():
        total = row.sum()
        print(f"    {prompt_labels.get(p,p):<18}: Hindi={row['Hindi']}({100*row['Hindi']/total:.0f}%) "
              f"English={row['English']}({100*row['English']/total:.0f}%) "
              f"Other={row['Other']}({100*row['Other']/total:.0f}%)")
    
    # Chi-square on language ~ prompt
    if len(pivot) >= 2:
        try:
            chi2, p_val, dof, _ = chi2_contingency(pivot.values)
            sig = "✓ Significant" if p_val < 0.05 else "✗ Not significant"
            print(f"  Chi-square (language ~ prompt): χ²={chi2:.2f}, df={dof}, p={p_val:.4f} → {sig}")
        except Exception as e:
            print(f"  Chi-square failed: {e}")

print("\n Interpretation:")
print("  Significant p < 0.05 means the model's language of response CHANGES")
print("  meaningfully across prompt conditions (e.g., P1 more Hindi, P4 more English).")
print("  P4 (Human Simulation) may produce more culturally specific (Hindi) responses.")

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 10: LLM × Human Convergence by Semantic Category
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("ANALYSIS 10: LLM × Human Convergence by Semantic Category")
print("(% of times LLM response matches MODAL human response per cue)")
print("═"*70)

# Modal human response per cue
modal_human = {}
for cue, grp in human_df.groupby('cue'):
    modal = grp['response_norm'].value_counts().idxmax()
    modal_human[cue] = modal

print(f"\n  Modal human responses:")
for cue, modal in sorted(modal_human.items()):
    print(f"    {cue:<16}: '{modal}'")

print("\n  Convergence with modal human response (by semantic category):")

for model in ['ChatGPT','Claude','Gemini']:
    print(f"\n  Model: {model}")
    print(f"  {'Category':<12} {'Converge':<10} {'Total':<8} {'%'}")
    print(f"  {'-'*40}")
    
    for cat in ['concrete','abstract','emotional','cultural']:
        sub = llm_df[(llm_df['model']==model) & (llm_df['sem_cat']==cat)]
        if sub.empty: continue
        matches = sum(
            row['response'].strip().lower() == modal_human.get(row['cue'],'__NONE__')
            for _, row in sub.iterrows()
        )
        total = len(sub)
        pct = 100*matches/total if total else 0
        print(f"  {cat:<12} {matches:<10} {total:<8} {pct:.1f}%")
    
    # Overall
    all_matches = sum(
        row['response'].strip().lower() == modal_human.get(row['cue'],'__NONE__')
        for _, row in llm_df[llm_df['model']==model].iterrows()
    )
    all_total = len(llm_df[llm_df['model']==model])
    print(f"  {'OVERALL':<12} {all_matches:<10} {all_total:<8} {100*all_matches/all_total:.1f}%")

print("\n Hypothesis: LLMs converge most on CONCRETE nouns (concrete things have")
print("  universal associations), least on CULTURAL words (Sanskar, Aashirwad).")

# Per-cue convergence table
print("\n  Per-cue convergence (all models pooled):")
print(f"  {'Cue':<16} {'Lang':<8} {'Category':<12} {'Modal':<14} {'Match%'}")
print(f"  {'-'*65}")
for cue in word_order:
    modal = modal_human.get(cue,'?')
    sub = llm_df[llm_df['cue']==cue]
    if sub.empty: continue
    matches = sum(r.strip().lower()==modal for r in sub['response'])
    pct = 100*matches/len(sub)
    print(f"  {cue:<16} {cue_language[cue]:<8} {semantic_category[cue]:<12} {modal:<14} {pct:.1f}%")

# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS 11: Between-Model Consistency
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("ANALYSIS 11: Between-Model Consistency (Pairwise Agreement)")
print("═"*70)

models = ['ChatGPT','Claude','Gemini']
pairs = [('ChatGPT','Claude'),('ChatGPT','Gemini'),('Claude','Gemini')]

def pairwise_agreement(m1, m2, df):
    """% of (prompt, cue) pairs where both models gave same response"""
    d1 = df[df['model']==m1].set_index(['prompt','cue'])['response'].str.lower()
    d2 = df[df['model']==m2].set_index(['prompt','cue'])['response'].str.lower()
    common = d1.index.intersection(d2.index)
    if len(common) == 0: return 0, 0
    matches = sum(d1[idx] == d2[idx] for idx in common)
    return matches, len(common)

print("\n  Pairwise LLM–LLM agreement:")
print(f"  {'Pair':<25} {'Matches':<10} {'Total':<8} {'Agreement %'}")
print(f"  {'-'*55}")
llm_llm_agreements = {}
for m1, m2 in pairs:
    m, t = pairwise_agreement(m1, m2, llm_df)
    pct = 100*m/t if t else 0
    llm_llm_agreements[(m1,m2)] = pct
    print(f"  {m1+' vs '+m2:<25} {m:<10} {t:<8} {pct:.1f}%")

# LLM vs Human: compute as % of LLM responses that appear in human set per cue
print("\n  LLM–Human agreement (% LLM responses found in human response set):")
print(f"  {'Model':<12} {'Matches':<10} {'Total':<8} {'Agreement %'}")
print(f"  {'-'*45}")
llm_human_agreements = {}
for model in models:
    sub = llm_df[llm_df['model']==model]
    matches = sum(
        row['response'].strip().lower() in human_resp_by_cue.get(row['cue'],set())
        for _, row in sub.iterrows()
    )
    total = len(sub)
    pct = 100*matches/total
    llm_human_agreements[model] = pct
    print(f"  {model:<12} {matches:<10} {total:<8} {pct:.1f}%")

avg_llm_llm = np.mean(list(llm_llm_agreements.values()))
avg_llm_human = np.mean(list(llm_human_agreements.values()))

print(f"\n  Average LLM–LLM agreement:   {avg_llm_llm:.1f}%")
print(f"  Average LLM–Human agreement: {avg_llm_human:.1f}%")

if avg_llm_llm > avg_llm_human:
    print("\n  ⚠️  LLMs agree with EACH OTHER more than with humans.")
    print("  This suggests shared training-data bias independent of human cognition.")
else:
    print("\n  ✓ LLMs do not systematically agree more with each other than with humans.")

# By cue for richer analysis
print("\n  Per-cue: do all 3 LLMs give the same response? (across all prompts)")
print(f"  {'Cue':<16} {'Total comparisons':<20} {'All-3 agree %'}")
print(f"  {'-'*50}")
for cue in word_order:
    sub = llm_df[llm_df['cue']==cue]
    by_prompt = sub.groupby('prompt')['response'].apply(lambda x: x.str.lower().tolist())
    all_agree = 0
    total = 0
    for prompt_resps in by_prompt:
        if len(prompt_resps) == 3:
            total += 1
            if len(set(prompt_resps)) == 1:
                all_agree += 1
    pct = 100*all_agree/total if total else 0
    print(f"  {cue:<16} {total:<20} {pct:.1f}%")

print("\n" + "═"*70)
print("ANALYSIS COMPLETE")
print("═"*70)
