# Prompts



---



Claude kreiraj teška pitanja za RAG
===================================

### Context
This prompt enables Claude (or another LLM) to create challenging evaluation questions from any documentation or FAQ document. These questions are specifically designed to test smaller LLM models in RAG (Retrieval-Augmented Generation) systems.

### The Prompt

```
# Task: Generate 10 Hard Questions for RAG Model Evaluation

You are given a source document (FAQ, documentation, or knowledge base). Your task is to create 10 challenging questions designed to evaluate smaller LLM models in a RAG (Retrieval-Augmented Generation) system.

## What Makes a Question "Hard"?

Hard questions should test one or more of these aspects:

### 1. Multi-hop Retrieval
- Answer requires synthesizing information from 2-3 different sections of the document
- Information is scattered across multiple Q&A pairs or paragraphs
- Requires connecting related concepts from different parts of the document

### 2. Conditional/Branching Logic
- Answer has IF-THEN-ELSE structure
- Multiple solution paths depending on conditions
- "If X then do Y, otherwise do Z" patterns

### 3. Comparison and Contrast
- Requires understanding differences between similar concepts
- Comparing trade-offs between multiple approaches
- Subtle distinctions in behavior or features

### 4. Embedded or Scattered Information
- Answer to question X is hidden within the answer to question Y
- Same information mentioned in 2-3 different places with slightly different context
- Requires aggregating partial information from multiple locations

### 5. Permission Boundaries and Limitations
- Understanding what users CAN'T do (negative information)
- Role-based permissions (who can do what)
- Technical limitations and constraints

### 6. Procedural Complexity
- Multi-step workflows (5+ steps)
- Procedures with critical warnings (data loss, security implications)
- Multiple alternative methods to achieve the same goal

### 7. Reasoning About "Why"
- Not just "what" or "how" but "why"
- Understanding underlying reasons or policies
- Security, performance, or design rationale

### 8. Technical Nuance
- Subtle technical distinctions
- Understanding system behavior at a deeper level
- Workarounds that have their own limitations

## Output Format

For each of the 10 questions, provide:

---

## Question N: [Short Descriptive Title] ([Difficulty Type])

**Question:**
[The actual question text in the source document's language]

**Segments needed:**

### Segment 1 ([Brief description])
**Location in document:** "[Title or heading of the section where this segment appears]"

**Text:**
```
[Exact text from the source document]
```

### Segment 2 (if needed)
**Location in document:** "[Title or heading]"

**Text:**
```
[Exact text from the source document]
```

[... additional segments as needed ...]

**Proposed Answer:**
[A reference answer synthesizing the information from all segments - should be comprehensive but concise, 2-4 sentences typically]

**Why this is hard:**
- [Bullet point 1: specific challenge this question poses]
- [Bullet point 2: retrieval difficulty]
- [Bullet point 3: reasoning complexity]
- [Bullet point 4: what makes it challenging for smaller models]

---

## Guidelines for Question Selection

### DO Select Questions That:
1. ✅ Require 2-3 document segments to answer completely
2. ✅ Have long, multi-paragraph answers (3+ paragraphs)
3. ✅ Include multiple solution paths or alternative approaches
4. ✅ Involve conditional logic (if this, then that)
5. ✅ Ask about differences between similar concepts or methods
6. ✅ Require understanding permissions or limitations
7. ✅ Have answers scattered across different sections
8. ✅ Include critical warnings about data loss, security, etc.
9. ✅ Ask "why" questions requiring reasoning beyond facts
10. ✅ Involve technical nuances that are easy to misunderstand

### DON'T Select Questions That:
1. ❌ Have simple, one-sentence answers
2. ❌ Are answered in a single, short paragraph
3. ❌ Ask for basic definitions or simple facts
4. ❌ Can be answered with yes/no without explanation
5. ❌ Have answers that are obvious from a single sentence
6. ❌ Are purely informational without complexity

## Distribution of Difficulty Types

Try to include at least one question from each category:
- 2-3 questions: Multi-hop retrieval (2-3 segments)
- 2-3 questions: Conditional/branching logic
- 1-2 questions: Comparison and contrast
- 1-2 questions: Permission boundaries/limitations
- 1-2 questions: Procedural complexity (multi-step)
- 1 question: Technical nuance or "why" reasoning

## Example of a Good Hard Question

**Question:**
"What should I check if users see a message that they don't have access to the forum because the 'reply' option is not visible?"

**Why this is a good hard question:**
- Multi-paragraph answer with 3+ paragraphs of explanation
- Conditional logic: If forced group mode is enabled, then do X, otherwise do Y
- Multiple solution paths: 3 different approaches to solve the problem
- Nested settings: Requires understanding interaction between course-level and activity-level settings
- Procedural complexity: Involves navigating through Settings > Edit Settings > specific options
- Technical understanding: Need to understand group visibility modes and their implications

## Step-by-Step Process

1. **Read the entire source document** to understand its structure and content

2. **Identify complex topics** that span multiple sections or have detailed explanations

3. **Look for answers with**:
   - Multiple paragraphs
   - Step-by-step procedures
   - Conditional statements (if/then)
   - Comparisons between options
   - References to other parts of the document

4. **For each potential question**:
   - Verify it requires 2-3 segments to answer fully
   - Check that the answer has meaningful complexity
   - Ensure it tests reasoning, not just recall

5. **Extract segments carefully**:
   - Copy exact text from the document
   - Note the section/heading where it appears
   - Include enough context for the segment to make sense

6. **Write a reference answer** that:
   - Synthesizes information from all segments
   - Is comprehensive but concise
   - Could serve as a gold standard for evaluation

7. **Explain why it's hard** with specific details about:
   - Retrieval challenges (how many segments, where located)
   - Reasoning required (conditional logic, comparison, etc.)
   - What makes it challenging for smaller models

## Quality Checklist

Before finalizing your 10 questions, verify:

- ✅ Each question requires at least 2 segments (at least 5 out of 10 questions)
- ✅ At least 3 questions involve conditional or branching logic
- ✅ At least 2 questions require comparing/contrasting different approaches
- ✅ At least 1 question asks about permissions or limitations
- ✅ All proposed answers are 2+ sentences and comprehensive
- ✅ All "Why this is hard" sections have 3-4 specific bullet points
- ✅ Questions cover different topics (not all about the same feature)
- ✅ Segments are exactly quoted from the source (no paraphrasing)
- ✅ Location references are accurate and specific

## Examples of Difficulty Type Labels

Use these labels in your question titles:
- `(Multi-hop, Multiple Workflows)`
- `(Scattered Info, Security Reasoning)`
- `(Subtle Behavioral Differences)`
- `(Multiple Procedures)`
- `(Permission Boundaries)`
- `(Conditional Workflow)`
- `(Trade-offs Between Approaches)`
- `(Technical Limitations)`
- `(Multi-step, Conditional Logic)`

## Important Reminders

1. **Preserve original language**: If the source document is in Croatian, German, French, etc., keep questions and segments in that language
2. **Exact quotations**: Copy segment text exactly, don't paraphrase
3. **Accurate locations**: Note the specific section/heading/question title where each segment appears
4. **Comprehensive answers**: Reference answers should synthesize all segments, not just repeat one
5. **Specific difficulty explanations**: In "Why this is hard", be specific about what makes it challenging (don't just say "it's complex")

## Final Output Structure

Your output should be structured as:

```markdown
# [Document Name] - 10 Hard Questions for Model Evaluation

This file contains 10 challenging questions designed to test smaller LLM models in a RAG context. Each question requires complex reasoning, multi-hop retrieval, or synthesis from multiple document segments.

---

## Question 1: [Title] ([Difficulty Type])
[Full question structure as specified above]

---

## Question 2: [Title] ([Difficulty Type])
[Full question structure as specified above]

---

[... continue for all 10 questions ...]

---

## Evaluation Criteria for These Questions

These 10 questions are designed to test:

1. **Multi-hop retrieval**: Questions requiring information from 2-3 different document sections
2. **Conditional reasoning**: If X then Y, else Z patterns
3. **Comparison/contrast**: Understanding differences between similar concepts
4. **Permission/limitation understanding**: What can vs cannot be done
5. **Procedural complexity**: Multi-step workflows with important warnings
6. **Synthesis**: Combining scattered information into coherent answer
7. **Technical reasoning**: Understanding WHY not just WHAT
8. **Negative information**: Understanding what's not possible/not allowed

## Difficulty Factors

- **Retrieval difficulty**: 1-3 segments needed, sometimes from different Q&A pairs
- **Answer embedded**: Some answers hidden within different questions
- **Length**: Answers range from 2 sentences to multiple paragraphs
- **Conditional logic**: Many answers have IF/ELSE branches
- **Trade-offs**: Several questions require understanding pros/cons of different approaches
```

---

Now, please read the provided source document and create 10 hard questions following this template and all guidelines above.
```

### Usage Instructions

1. Copy the prompt above (everything between the triple backticks)
2. Provide it to Claude along with your source document
3. Claude will analyze the document and generate 10 hard questions with:
   - The question text
   - Exact segments from the source document needed to answer it
   - Location references for each segment
   - A proposed reference answer
   - Explanation of why each question is challenging

### Expected Output

The output will be a markdown file with 10 questions suitable for evaluating:
- Retrieval accuracy (can the system find the right 2-3 segments?)
- Generation quality (can the model synthesize a coherent answer?)
- Reasoning capability (can smaller models handle conditional logic, comparisons, etc.?)

### Example Use Case

This prompt was used to generate `loomen_faq.hard-questions.md` from the Loomen FAQ documentation, creating 10 questions that test multi-hop retrieval, conditional reasoning, permission boundaries, and procedural complexity.

```



---



Prompt za LLM odgovaranje iz cijelog dokumenta
==============================================

### Context
This prompt is designed for testing smaller LLMs' ability to comprehend and answer questions when the ENTIRE source document is available in the context window. This tests:
- Handling large context windows (10k+ tokens)
- Finding relevant information within a long document (needle-in-haystack)
- Synthesizing information from distant parts of the document
- Maintaining focus despite many similar-looking sections

The prompt is structured to maximize **prompt caching**: the document and instructions are static and cacheable, while only the question changes between requests.

### The Prompt

```
# Task: Answer Question Using Full Document Context

You are a helpful assistant that answers questions based on a complete documentation or FAQ document. The entire document is provided below. Your goal is to provide accurate, comprehensive answers using ONLY the information available in the document.

## Guidelines for Answer Generation

### DO:
1. ✅ **Answer in the same language** as the question (Croatian, English, etc.)
2. ✅ **Search the ENTIRE document** - relevant information might be anywhere
3. ✅ **Base your answer ONLY on the document** - do not use external knowledge
4. ✅ **Combine information from multiple sections** when the answer is scattered
5. ✅ **Be thorough** - check all parts of the document before finalizing your answer
6. ✅ **Preserve important details**: URLs, emails, phone numbers, procedures, warnings, conditional logic
7. ✅ **Use clear structure** for complex answers with paragraphs, lists, or numbered steps
8. ✅ **Acknowledge multiple solutions** if the document presents different approaches
9. ✅ **Cross-reference related information** from different parts of the document

### DON'T:
1. ❌ **Don't stop searching** after finding the first mention - there may be more complete information elsewhere
2. ❌ **Don't add information** not present in the document
3. ❌ **Don't make assumptions** beyond what the document states
4. ❌ **Don't say "according to the document"** - just answer directly
5. ❌ **Don't skip important details** like URLs, contact information, or warnings
6. ❌ **Don't ignore related sections** even if they're far from the main answer
7. ❌ **Don't change the language** of the answer from the question's language

## Special Considerations

**Information may be scattered**: The same topic might be mentioned in multiple places with varying detail. Look for the MOST COMPLETE explanation.

**Related content might be distant**: In FAQ documents, related questions might be separated by many unrelated questions.

**Completeness check**: Before finalizing, verify you've checked the entire document for relevant information.

## Handling Insufficient Information

If the document does NOT contain enough information:
- Croatian: "U dostupnom dokumentu nema dovoljno informacija za potpun odgovor na ovo pitanje."
- English: "The provided document does not contain enough information to fully answer this question."
- Provide whatever partial information IS available
- Do NOT make up information

## Quality Checklist

Before answering, verify:
- ✅ Searched the ENTIRE document, not just first matches
- ✅ Answer is in the same language as the question
- ✅ All relevant information from different sections is included
- ✅ Important details preserved
- ✅ Structure is clear
- ✅ No external knowledge added

---

<document>
[COMPLETE DOCUMENT CONTENT GOES HERE]
</document>

<question>
[USER'S QUESTION GOES HERE]
</question>
```

### Usage Instructions

1. **Structure for prompt caching**:
   - Everything before `<question>` is **static and cacheable**
   - Only the `<question>` content changes between requests
   - This dramatically reduces costs and latency for multiple questions

2. **Insert content**:
   ```python
   # Read the complete document once
   with open('loomen_faq.md', 'r', encoding='utf-8') as f:
       full_document = f.read()

   # Build the static prompt (cache this!)
   static_prompt = f"""
   [All the instructions from above]

   <document>
   {full_document}
   </document>

   """

   # For each question, append only this part
   question = "Koje ovlasti ima administrator kategorije?"
   full_prompt = static_prompt + f"<question>\n{question}\n</question>"

   # Send to LLM
   answer = llm.generate(full_prompt)
   ```

3. **Caching benefits**:
   - First request: Processes ~20k tokens (instructions + document + question)
   - Subsequent requests: Only processes ~50 tokens (the question)
   - Cost reduction: ~99% for cached portion
   - Latency reduction: Significantly faster responses

### Example API Call with Caching

```python
from anthropic import Anthropic

client = Anthropic(api_key="your-api-key")

# Read document once
with open('loomen_faq.md', 'r', encoding='utf-8') as f:
    document_content = f.read()

# Build static system prompt (this will be cached)
system_prompt = """
[The full instructions from above]

<document>
{document_content}
</document>
""".format(document_content=document_content)

# Ask multiple questions - only the user message changes
questions = [
    "Koje ovlasti ima administrator kategorije?",
    "Kako mogu prebaciti tečaj sa Merlina na Loomen?",
    "Mogu li učenici otvarati tečajeve?"
]

for question in questions:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}  # Cache this!
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"<question>\n{question}\n</question>"
            }
        ]
    )
    print(f"Q: {question}")
    print(f"A: {response.content[0].text}\n")
```

### Context Window Requirements

- **Small docs (< 5k tokens)**: Any model with 8k+ context window
- **Medium docs (5k-20k tokens)**: Models with 32k+ context window
- **Large docs (20k-50k tokens)**: Models with 128k+ context window
- **Very large docs (> 50k tokens)**: Models with 200k+ context window

**Loomen FAQ example**: ~15-20k tokens, works well with 32k+ context models

### Expected Output Quality

The LLM should:
- Find scattered information across the entire document
- Synthesize multi-location answers coherently
- Preserve all details from the original document
- Handle needle-in-haystack questions effectively

### Comparison: Full Document vs RAG

**Full Document Approach:**
- ✅ Never misses information (everything available)
- ✅ Can synthesize from distant sections
- ✅ Simpler architecture (no retrieval needed)
- ✅ With caching: Very efficient for multiple questions
- ❌ Requires large context window
- ❌ Without caching: More expensive per query
- ❌ May struggle with very long documents

**RAG with Retrieved Chunks:**
- ✅ Works with smaller context windows
- ✅ Scales to massive knowledge bases
- ❌ Dependent on retrieval quality
- ❌ May miss scattered information
- ❌ More complex architecture

### Use Cases

This approach is ideal for:
- **Model evaluation**: Testing comprehension with large contexts
- **Baseline comparison**: Comparing against RAG retrieval quality
- **Small knowledge bases**: When entire KB fits in context
- **Multiple questions on same document**: Caching makes this very efficient
- **Critical applications**: Where missing information is unacceptable

### Evaluation Metrics

Measure:
1. **Answer accuracy**: Factual correctness
2. **Answer completeness**: Found ALL relevant information?
3. **Synthesis quality**: Combined scattered info well?
4. **Position bias**: Quality consistent regardless of info location?
5. **Distractor resistance**: Ignores irrelevant similar sections?

```

---



Prompt za LLM za odgovaranje iz RAG konteksta
==============================================

### Context
This prompt is designed for the RAG pipeline where an LLM receives a user question along with retrieved context chunks from a knowledge base. The LLM must synthesize an accurate, helpful answer based solely on the provided context.

The prompt is structured to enable **partial prompt caching**: the instructions are static and cacheable, while the context and question change with each request.

### The Prompt

```
# Task: Answer Question Based on Retrieved Context

You are a helpful assistant that answers questions based on provided context from a knowledge base. Your goal is to provide accurate, comprehensive answers using ONLY the information available in the context.

## Guidelines for Answer Generation

### DO:
1. ✅ **Answer in the same language** as the question (Croatian, English, etc.)
2. ✅ **Base your answer ONLY on the provided context** - do not use external knowledge
3. ✅ **Synthesize information** from multiple context segments when needed
4. ✅ **Be comprehensive** - include all relevant details from the context
5. ✅ **Preserve important details**: URLs, emails, phone numbers, procedures, warnings, conditional logic, alternative approaches
6. ✅ **Use clear structure** for complex answers with paragraphs, lists, or numbered steps
7. ✅ **Be direct and concise** while remaining complete
8. ✅ **Acknowledge multiple solutions** if the context presents different approaches

### DON'T:
1. ❌ **Don't add information** not present in the context
2. ❌ **Don't make assumptions** beyond what the context states
3. ❌ **Don't say "based on the context"** or "according to the provided information" - just answer directly
4. ❌ **Don't skip important details** like URLs, contact information, or warnings
5. ❌ **Don't oversimplify** complex procedures that have multiple steps
6. ❌ **Don't ignore** any of the provided context segments
7. ❌ **Don't change the language** of the answer from the question's language

## Handling Insufficient Context

If the context does NOT contain enough information:
- Croatian: "Na temelju dostupnih informacija ne mogu pružiti potpun odgovor na ovo pitanje."
- English: "I don't have enough information in the provided context to fully answer this question."
- Provide whatever partial information IS available
- Do NOT make up information or use knowledge outside the context

## Quality Checklist

Before answering, verify:
- ✅ Answer is in the same language as the question
- ✅ All information comes from the provided context
- ✅ Important details (URLs, emails, steps) are included
- ✅ Answer directly addresses the question
- ✅ Structure is clear and easy to follow
- ✅ No external knowledge or assumptions added
- ✅ For multi-step procedures, steps are in correct order
- ✅ For conditional answers, all branches are explained

---

<context>
[RETRIEVED CONTEXT SEGMENTS GO HERE - can be multiple segments separated by ---]
</context>

<question>
[USER'S QUESTION GOES HERE]
</question>
```

### Usage Instructions

1. **Structure for prompt caching**:
   - **Static/Cacheable**: All instructions (everything before `<context>`)
   - **Dynamic**: `<context>` and `<question>` content (changes per request)
   - The instructions can be cached, reducing costs by ~50% per request

2. **Insert content**:
   ```python
   # Build the static prompt (cache this!)
   static_instructions = """
   [All the instructions from above, up to the --- line]
   """

   # For each RAG query:
   retrieved_context = retriever.get_context(question)

   full_prompt = static_instructions + f"""
   <context>
   {retrieved_context}
   </context>

   <question>
   {question}
   </question>
   """

   answer = llm.generate(full_prompt)
   ```

3. **Multiple context segments**:
   If your retriever returns multiple chunks, combine them:
   ```python
   context_chunks = retriever.get_top_k(question, k=3)
   combined_context = "\n\n---\n\n".join(context_chunks)
   ```

### Example API Call with Caching

```python
from anthropic import Anthropic

client = Anthropic(api_key="your-api-key")

# Static system prompt (cache this!)
system_prompt = """
# Task: Answer Question Based on Retrieved Context

[All the instructions from above]

---
"""

# For each RAG query:
question = "Koje ovlasti ima administrator kategorije?"
retrieved_context = """
Administrator kategorije (menadžer) može pristupiti svim tečajevima u kategoriji
ustanove za koju je imenovan, može kreirati nove tečajeve brisati postojeće tečajeve
iz kategorije. Administrator kategorije ima ujedno i sve ovlasti predavača unutar
dodijeljene kategorije.
"""

response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"}  # Cache the instructions!
        }
    ],
    messages=[
        {
            "role": "user",
            "content": f"""<context>
{retrieved_context}
</context>

<question>
{question}
</question>"""
        }
    ]
)

print(response.content[0].text)
```

### Caching Benefits

- **First request**: Processes instructions (~1k tokens) + context + question
- **Subsequent requests**: Only processes context + question (~500-2k tokens)
- **Cost reduction**: ~30-50% depending on context size
- **Latency reduction**: Faster response times

### Answer Structure Examples

**Simple question** (direct answer):
```
Question: "Koji je elektronički identitet potreban za prijavu?"
Answer: "Za prijavu na sustav Loomen za škole potreban je AAI@Edu.hr elektronički
identitet izdan na školi. Korisnici mogu koristiti gumb AAI@Edu.hr ili Škole LDAP."
```

**Multi-step procedure** (structured answer):
```
Question: "Kako mogu prebaciti tečaj sa Merlina na Loomen?"
Answer: "U prvom koraku potrebno je od administratora sustava Merlin zatražiti
sigurnosne kopije tečajeva. Ako želite sačuvati korisničke unose, zatražite kopije
koje uključuju podatke o korisnicima.

Kopije tečajeva dostavite administratorima sustava Loomen putem e-pošte
loomen@carnet.hr. Ako podaci o korisnicima nisu potrebni, možete zatražiti sigurnosnu
kopiju bez korisničkih podataka i sami je obnoviti u Loomenu za škole."
```

**Conditional logic** (clear branches):
```
Question: "Što učiniti ako forum ne pokazuje opciju odgovori?"
Answer: "Ako je u postavkama učionice postavljeno obavezno korištenje odvojenih
grupa, potrebno je za svaku grupu kreirati novo pitanje unutar foruma uz odabir
željene grupe.

Alternativno rješenje je promijeniti postavke tečaja: u bloku Postavke > Uredi
postavke odaberite 'Ne' za 'Obveži na grupni oblik nastave'. Dodatno, u postavkama
foruma odaberite 'Vidljive grupe' ili 'Bez grupa'."
```

### Expected Output Quality

The LLM should generate answers that:
- **Are factually accurate** (grounded in the provided context)
- **Are appropriately detailed** (comprehensive but not verbose)
- **Follow proper structure** (clear paragraphs, lists when needed)
- **Preserve critical information** (URLs, procedures, warnings)
- **Match the question's language** (Croatian for Croatian questions)

### Use Cases

This prompt is suitable for:
- **RAG chatbot systems** answering user questions
- **FAQ assistance systems** with semantic search
- **Documentation Q&A interfaces** with vector retrieval
- **Knowledge base query systems** with hybrid search
- **LLM evaluation** in RAG pipelines (answer generation quality)

### Comparison with Full Document Approach

**RAG with Retrieved Context:**
- ✅ Works with smaller context windows (2k-8k tokens typical)
- ✅ Faster inference (less tokens to process)
- ✅ Lower cost per query
- ✅ Scales to massive knowledge bases (millions of documents)
- ❌ Quality depends on retrieval accuracy
- ❌ May miss scattered information if retrieval fails
- ❌ More complex architecture (embedding + vector DB + retrieval)

**Full Document in Context:**
- ✅ Never misses information (everything available)
- ✅ Can synthesize from distant sections
- ✅ Simpler architecture
- ❌ Requires large context windows (32k-200k tokens)
- ❌ More expensive without caching
- ❌ Doesn't scale to large knowledge bases

### Evaluation Metrics

When evaluating this RAG approach, measure:
1. **Answer accuracy**: Factual correctness given the context
2. **Answer completeness**: Uses all relevant info from context
3. **Hallucination rate**: Adding info not in context
4. **Synthesis quality**: Combines multiple context segments well
5. **Format adherence**: Follows structure guidelines
6. **Language consistency**: Matches question language

```




