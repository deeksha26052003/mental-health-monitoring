# knowledge_base.py
# Small local knowledge base of coping-strategy / resource snippets.
# Each entry is (tag, text). The tag helps you see which risk factor each
# snippet is meant to address; it is not used by the retrieval itself.

DOCUMENTS = [
    ("stress", "Regular short breaks during the day, even five minutes, "
               "can reduce sustained stress levels. Simple breathing "
               "exercises (slow inhale for 4 seconds, hold for 4, exhale "
               "for 6) activate the parasympathetic nervous system and "
               "lower physiological stress markers."),

    ("sleep", "Irregular sleep schedules are linked to higher anxiety and "
              "depression risk. Keeping a consistent wake-up time, even on "
              "weekends, is one of the more effective ways to stabilize "
              "mood and energy over a few weeks."),

    ("exercise", "Even light, regular movement (a 20-minute walk most days) "
                 "is associated with meaningfully lower depression "
                 "screening scores compared to no activity at all. "
                 "Consistency matters more than intensity."),

    ("work_life_balance", "Poor work-life balance is one of the strongest "
     "predictors of burnout-related depressive "
     "symptoms. Setting one small, protected block of "
     "personal time each day (even 30 minutes) has "
     "been shown to buffer against this."),

    ("panic", "During a panic attack, grounding techniques such as the "
              "5-4-3-2-1 method (naming 5 things you see, 4 you can touch, "
              "3 you hear, 2 you smell, 1 you taste) can help interrupt the "
              "acute physiological spiral."),

    ("family_history", "Having a family history of mental illness increases "
     "baseline risk, but it is not deterministic. Early, "
     "proactive support (therapy, regular check-ins) has "
     "been shown to meaningfully reduce the gap between "
     "genetic risk and actual outcomes."),

    ("substance_use", "Frequent substance use as a coping mechanism tends "
     "to worsen underlying anxiety and depression over "
     "time, even when it provides short-term relief. "
     "Replacing it gradually with another regulation "
     "strategy (exercise, social contact, therapy) tends "
     "to have better long-term outcomes than stopping "
     "abruptly without support."),

    ("screen_time", "High daily screen time, especially late at night, is "
                    "associated with poorer sleep quality, which in turn "
                    "worsens mood regulation. A wind-down period without "
                    "screens in the last 30-60 minutes before bed is a "
                    "commonly recommended first step."),

    ("general_support", "Speaking with a licensed mental health "
     "professional is the most effective way to get an "
     "accurate, individualized understanding of "
     "symptoms. Screening tools and models like this "
     "one can flag risk, but they cannot diagnose."),

    ("crisis", "If you are in crisis or having thoughts of self-harm, "
               "please contact a crisis line immediately: in the US, call "
               "or text 988 (Suicide & Crisis Lifeline), available 24/7."),
]
