const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');

require('dotenv').config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  timeout: 5000,
});

const settings = {
  model: "text-embedding-ada-002",
  // dimensions: 1536,
};

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
const getEmbedding = async (text, cache) => {
    if (cache[text]) {
        return cache[text];
    } else {
        const embeddingResponse = await openai.embeddings.create({
            ...settings,
            input: text,
        });

        const embedding = embeddingResponse.data[0]?.embedding;

        cache[text] = embedding;
        saveCache(cache);
        return embedding;
    }
};

// Calculate Euclidean distance
const euclideanDistance = (vec1, vec2) => {
    return Math.sqrt(vec1.reduce((sum, value, index) => sum + (value - vec2[index]) ** 2, 0));
};

// Main function to compare texts
const compareTexts = async (text1, text2) => {
    const cache = loadCache();
    const embedding1 = await getEmbedding(text1, cache);
    const embedding2 = await getEmbedding(text2, cache);
    const distance = euclideanDistance(embedding1, embedding2);
    console.log(`Euclidean Distance between '${text1}' and '${text2}': ${distance}`);
};

// Example texts
const text1 = `Has a driver's license. Knows C#. Has a driver's license. Knows C#. Has a driver's license. Knows C#. Has a driver's license. Knows C#`;
const text2 = `Knows C#. Has driver's license. Knows C#. Has driver's license. Knows C#. Has driver's license. Knows C#. Has driver's license`;

compareTexts(text1, text2);
