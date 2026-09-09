// Markdown rendering, and the escaping that has to hold underneath it.
//
//   node tests/markdown.test.mjs
//
// A room is shared: the person reading an answer is often not the person who wrote
// the prompt that produced it. Model output is therefore untrusted input, and the
// injection tests below matter as much as the formatting ones.

import { render, renderStreaming } from "../room/markdown.js";

let pass = 0, fail = 0;
const ok = (name, cond, extra = "") => {
  if (cond) { pass++; console.log("  ok   " + name); }
  else { fail++; console.log("  FAIL " + name + (extra ? "\n         " + extra : "")); }
};

console.log("\nformatting");
ok("paragraphs", render("hello there") === "<p>hello there</p>");
ok("bold", render("a **b** c").includes("<strong>b</strong>"));
ok("italic", render("a *b* c").includes("<em>b</em>"));
ok("inline code", render("use `npm test` now").includes("<code>npm test</code>"));
ok("headings shift down (## -> h4)", render("## Title").startsWith("<h4>"));
ok("bullet list", (() => {
  const h = render("- one\n- two");
  return h.includes("<ul>") && h.includes("<li>one</li>") && h.includes("<li>two</li>");
})());
ok("numbered list", (() => {
  const h = render("1. first\n2. second");
  return h.includes("<ol>") && h.includes("<li>first</li>");
})());
ok("blockquote", render("> quoted").includes("<blockquote>quoted</blockquote>"));
ok("horizontal rule", render("---").includes("<hr>"));

console.log("\ncode blocks");
{
  const h = render("```python\ndef f(x):\n    return x*2\n```");
  ok("fenced block becomes pre/code", h.includes("<pre") && h.includes("</code></pre>"));
  ok("language is recorded", h.includes('data-lang="python"'));
  ok("indentation survives", h.includes("    return x*2"));
  ok("markup inside code is not interpreted",
     render("```\n**not bold**\n```").includes("**not bold**"));
  ok("inline code is not interpreted either",
     render("`**not bold**`").includes("<code>**not bold**</code>"));
}

console.log("\nescaping and injection");
{
  const h = render("<script>alert(1)</script>");
  ok("script tags are escaped, not emitted", !h.includes("<script>") && h.includes("&lt;script&gt;"));
  ok("attribute breakout is escaped", !render('" onerror="x').includes('onerror="'));
  ok("ampersands survive", render("a & b").includes("&amp;"));
  ok("html in a code block is escaped too",
     render("```\n<img src=x onerror=alert(1)>\n```").includes("&lt;img"));

  const link = render("[click](https://example.com/x)");
  ok("http links render with noopener", link.includes('href="https://example.com/x"') && link.includes("noopener"));
  // The payload stays visible as inert text, which is the correct outcome: the point
  // is that no <a> is produced, not that the characters disappear.
  ok("javascript: links produce no anchor", (() => {
    const h2 = render("[click](javascript:alert(1))");
    return !h2.includes("<a ") && !h2.includes("href=");
  })());
  ok("data: links are refused", !render("[x](data:text/html,<script>)").includes("<a "));
}

console.log("\nstreaming");
{
  ok("an unclosed fence still renders as code",
     renderStreaming("```python\ndef f():").includes("<pre"));
  ok("a closed fence is unchanged by the streaming wrapper",
     renderStreaming("```\nx\n```") === render("```\nx\n```"));
  ok("empty input is safe", render("") === "" && renderStreaming("") === "");
  ok("null and undefined are safe", render(null) === "" && render(undefined) === "");
}

console.log("\na realistic answer");
{
  const answer = `Here is **two-sum**:

\`\`\`python
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
\`\`\`

Complexity:
- Time: \`O(n)\`
- Space: \`O(n)\``;
  const h = render(answer);
  ok("renders a mixed answer without leaking raw markup",
     h.includes("<strong>two-sum</strong>") && h.includes('data-lang="python"') &&
     h.includes("<ul>") && h.includes("<code>O(n)</code>") &&
     !h.includes("```") && !h.includes("**"));
}

console.log(`\n${pass} passed, ${fail} failed\n`);
process.exit(fail ? 1 : 0);
