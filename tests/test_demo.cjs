const assert = require('node:assert/strict');
const {test} = require('node:test');
const demo = require('../gemma_rlcd/static/demo-utils.js');
const config = require('../gemma_rlcd/static/visual-demo.json');
test('visual schemas contain distinct meaningful checks at each size', () => {
  for (const size of [32,64,128]) {
    const questions = demo.questions(config,size);
    assert.equal(Object.values(questions).reduce((n,q) => n + Object.keys(q.criteria).length,0),size);
    assert.equal(new Set(Object.values(questions).flatMap(q => Object.values(q.criteria))).size,size);
  }
  assert.throws(() => demo.questions(config,129));
});
test('partial JSON reveals only complete booleans with a delimiter', () => {
  const input = '{"people":{"person":true,"walking":false},"objects":{"car":true}}';
  assert.deepEqual({...demo.partialBooleans(input)}, {'people.person':true,'people.walking':false,'objects.car':true});
  const prefix = '{"people":{"person":true';
  assert.deepEqual({...demo.partialBooleans(prefix)}, {});
  assert.deepEqual({...demo.partialBooleans(prefix + ',"walking":fa')}, {'people.person':true});
  assert.deepEqual({...demo.partialBooleans('{"people":{"person":"true"}')}, {});
  assert.deepEqual({...demo.partialBooleans('{"people":{"person":truefake,')}, {});
  assert.deepEqual({...demo.partialBooleans('```json\n' + input + '\n```')}, {...demo.partialBooleans(input)});
});
