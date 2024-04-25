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
  const normalized = 1 / (1 + Math.exp(-dotProduct));
  return normalized;
}


const cache = loadCache();

// Main function to compare texts
const compareTexts = async (testCase, model, evalName, evalFunc) => {
    const embedding1 = await getEmbedding(testCase.first, cache, model);
    const embedding2 = await getEmbedding(testCase.second, cache, model);
    const distance = evalFunc(embedding1, embedding2);
    console.log(`${evalName}: ${distance}`);
    return distance;
};

const exampleTestCases = [
  {
    name: 'long-reverse-order',
    first: `Has a driver's license. Knows C#. Has a driver's license. Knows C#. Has a driver's license. Knows C#. Has a driver's license. Knows C#`,
    second: `Knows C#. Has driver's license. Knows C#. Has driver's license. Knows C#. Has driver's license. Knows C#. Has driver's license`
  },
  {
    name: 'reverse-order',
    first: `Has a driver's license. Knows C#.`,
    second: `Knows C#. Has driver's license.`
  },
  {
    name: 'has-vs-require',
    first: `Has B-driver license`,
    second: `Requires B-driver's license`
  },
  {
    name: 'fluent-vs-must',
    first: `Must speak Swedish and English`,
    second: `Fluent in Swedish and English`
  },
  {
    name: 'not-a-requirement',
    first: `Not a requirement to speak Swedish`,
    second: `Fluent in Swedish`
  }
]

if (!fs.existsSync('test-cases.json')) {
  fs.writeFileSync('test-cases.json', JSON.stringify(exampleTestCases, null, 2));
}

const testCases = JSON.parse(fs.readFileSync('test-cases.json', 'utf8'));

const testModels = [
  {
    model: "text-embedding-ada-002",
  },
  {
    model: "text-embedding-3-large",
    dimensions: 1536,
  },
  {
    model: "text-embedding-3-small"
  },
]

const evaluationModes = {
  'euclidean': euclideanDistance,
  'cosine': cosineDistance,
  'neg-inner-product': negativeInnerProduct,
  'manhattan': manhattanDistance
}

async function main() {
  const csvHeader = ['model', 'test-case', ...Object.keys(evaluationModes)];
  const csvRows = [];
  for (const model of testModels) {
    console.log('Evaluating model:', model.model);
    for (const testCase of testCases) {
      console.log('Test case:', testCase.name);
      const scores = []
      for (const evaluationMode of Object.keys(evaluationModes)) {
        const evaluationFunction = evaluationModes[evaluationMode];
        const score = await compareTexts(testCase, model, evaluationMode, evaluationFunction);
        scores.push(`${Math.round(score.toFixed(2) * 100)}%`);
      }
      csvRows.push([model.model, testCase.name, ...scores]);
    }
    console.log('---\n\n');
  }

  fs.writeFileSync('output.csv', csvHeader.join(',') + '\n' + csvRows.map(row => row.join(',')).join('\n'));
  console.log('Scores saved to output.csv')
};

main();