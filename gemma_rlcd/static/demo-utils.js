"use strict";

const VisualDemo = {
  questions(config, count) {
    if (![32, 64, 128].includes(count)) throw new Error("Choose 32, 64, or 128 checks.");
    return Object.fromEntries(config.groups.map((group) => [group.id, {
      type: "independent",
      instructions: "Which of these checks are visually established?",
      criteria: Object.fromEntries(group.checks.slice(0, count / config.groups.length).map((check) => [check.id, check.description])),
    }]));
  },
  partialBooleans(text) {
    // Read only completed boolean literals in the expected two-level JSON shape.
    // Partial strings, quoted booleans, and malformed suffixes are not answers.
    text = text.replace(/^\s*```(?:json)?\s*\n/, "");
    let index = 0;
    const values = Object.create(null);
    const skip = () => { while (/\s/.test(text[index] || "x")) index++; };
    const take = (char) => { skip(); if (text[index] !== char) return false; index++; return true; };
    const string = () => {
      skip();
      if (text[index] !== '"') return null;
      const start = index++;
      while (index < text.length) {
        if (text[index] === "\\") { index += 2; continue; }
        if (text[index++] === '"') {
          try { return JSON.parse(text.slice(start, index)); } catch { return null; }
        }
      }
      return null;
    };
    if (!take("{")) return values;
    while (index < text.length) {
      const group = string();
      if (group === null || !take(":") || !take("{")) break;
      while (index < text.length) {
        const key = string();
        if (key === null || !take(":")) return values;
        skip();
        const match = /^(true|false)(?=\s*[,}])/.exec(text.slice(index));
        if (!match) return values;
        values[`${group}.${key}`] = match[1] === "true";
        index += match[1].length;
        skip();
        if (take("}")) break;
        if (!take(",")) return values;
      }
      skip();
      if (take("}")) return values;
      if (!take(",")) return values;
    }
    return values;
  },
};
if (typeof module !== "undefined") module.exports = VisualDemo;
