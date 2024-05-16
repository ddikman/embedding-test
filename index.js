const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');

require('dotenv').config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  timeout: 5000,
});

const cacheFilePath = path.join(__dirname, '.cache.json');

// Load cache from file
const loadCache = () => {
    try {
        const data = fs.readFileSync(cacheFilePath, 'utf8');
        return JSON.parse(data);
    } catch (error) {
      console.error(error);
        return {};
    }
};

// Save cache to file
const saveCache = (cache) => {
    fs.writeFileSync(cacheFilePath, JSON.stringify(cache, null, 2));
};

// Get embedding from cache or API
const getEmbedding = async (text, cache, settings) => {
    if (cache[settings.model] && cache[settings.model][text]) {
        return cache[settings.model][text];
    } else {
        const embeddingResponse = await openai.embeddings.create({
            ...settings,
            input: text,
        });

        const embedding = embeddingResponse.data[0]?.embedding;

        if (!cache[settings.model]) {
          cache[settings.model] = {}
        }
        cache[settings.model][text] = embedding;
        saveCache(cache);
        return embedding;
    }
};


const euclideanDistance = (vec1, vec2) => {
    const euclidean = Math.sqrt(vec1.reduce((sum, value, index) => sum + (value - vec2[index]) ** 2, 0));
    const normalized = Math.exp(-euclidean);
    return normalized;
};

const manhattanDistance = (vec1, vec2) => {
  const manhattan = vec1.reduce((sum, value, index) => sum + Math.abs(value - vec2[index]), 0);
  const normalized = Math.exp(-manhattan);
  return normalized;
}

const cosineDistance = (vec1, vec2) => {
  const dotProduct = vec1.reduce((acc, val, i) => acc + val * vec2[i], 0);
  const magnitude1 = Math.sqrt(vec1.reduce((acc, val) => acc + val * val, 0));
  const magnitude2 = Math.sqrt(vec2.reduce((acc, val) => acc + val * val, 0));
  const cosine = dotProduct / (magnitude1 * magnitude2);
  const normalized =  (cosine + 1) / 2;
  return normalized;
}

const negativeInnerProduct = (vec1, vec2) => {
  const dotProduct = -vec1.reduce((acc, val, i) => acc + val * vec2[i], 0);
  const normalized = (1 + Math.exp(dotProduct)) ** -1;
  return normalized;
}


const cache = loadCache();

// Main function to compare texts
const compareTexts = async (text, term, model, evalName, evalFunc) => {
    const embedding1 = await getEmbedding(text, cache, model);
    const embedding2 = await getEmbedding(term, cache, model);
    const distance = evalFunc(embedding1, embedding2);
    return distance;
};

if (!fs.existsSync('test-cases.json')) {
  const exampleTestCases = {
    "terms": [ "rat", "hat", "cat", "building", "construction worker", "president", "cheese", "Bjorn Borg", "kitchenette", "makeup" ],
    "texts": [
      { "name": "The rat in a hat", "text": "The rat in a hat" }
    ]
  };

  fs.writeFileSync('test-cases.json', JSON.stringify(exampleTestCases, null, 2));
}

const testCases = JSON.parse(fs.readFileSync('test-cases.json', 'utf8'));

const testModel = {
  model: "text-embedding-3-large",
  dimensions: 1536,
}

const evaluationMethodName = 'cosine';
const evaluationMethod = cosineDistance;

async function main() {
  const csvHeader = ['model', 'evaluation', 'text-abbreviation', 'term', 'score'];
  const csvRows = [];

  const modelDescription = `${testModel.model}[${testModel.dimensions}]`
  for (const testCase of testCases.texts) {
    for (const term of testCases.terms) {
      const score = await compareTexts(testCase.text, term, testModel, evaluationMethodName, evaluationMethod);
      const scoreFormatted = `${Math.round(score.toFixed(2) * 100)}%`;
      csvRows.push([modelDescription, evaluationMethodName, `"${testCase.name}"`, `"${term}"`, scoreFormatted]);
      console.log(`[${testCase.name}] ${evaluationMethodName} with [${term}] = ${scoreFormatted}`);
    }
  }
  console.log('---\n\n');

  fs.writeFileSync('output.csv', csvHeader.join(',') + '\n' + csvRows.map(row => row.join(',')).join('\n'), { encoding: 'utf8', flag: 'w' });
  console.log('Scores saved to output.csv')
};

main();