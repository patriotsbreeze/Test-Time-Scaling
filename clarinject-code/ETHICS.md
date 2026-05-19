# Ethics Statement

## Nature of This Research

ClarInject-Code is **defensive security research**. Our goal is to surface a
vulnerability in coding agent interaction protocols so that practitioners can
build safer systems. We do not provide novel attack techniques; the injection
patterns we study are well-known in the adversarial-prompting literature. Our
contribution is the *measurement* of how those patterns are amplified by the
clarification interaction state, and the *mitigation* of that amplification.

## Scope of Harm

All adversarial payloads in this benchmark are:

1. **Synthetic and clearly labeled.** Each payload is annotated with its
   `attack_category` and is designed to be recognizable as a test case.
2. **Sandboxed.** The evaluation harness (`sandbox.py`) runs all agent code
   in an isolated temporary directory with no real network access. Exfiltration
   URLs use the reserved domain `http://example-sandbox.invalid`, which is
   unreachable by construction.
3. **Non-functional outside the sandbox.** No payload is designed to work if
   copied verbatim into a production environment without modification.

## What We Do Not Do

- We do not target real systems, APIs, or codebases.
- We do not release model weights or fine-tuned models.
- We do not publish jailbreak prompts designed to bypass model safety filters.

## Responsible Disclosure

If our benchmark reveals new attack vectors not previously documented, we will
follow coordinated disclosure with the relevant model providers before public
release of any such scenario.

## IRB / Ethics Review

This study involves no human subjects. All LLM API calls are made under
standard commercial terms. No personally identifiable information is collected
or processed.

## Broader Impact

We expect this work to have a net positive impact: by quantifying and
publicizing the clarification tax, we create pressure for model providers and
agent framework developers to implement provenance-aware input handling. The
mitigations we propose (Section 5) are lightweight and deployable today.
