let currentIndex = 0;
let imageData = {};

document.addEventListener('DOMContentLoaded', (event) => {
    fetchData();
    console.log(imageData)
});

async function fetchData() {
  const response = await fetch('/getData', {method: 'POST'});
  const data = await response.json()
  fetchImage(0); // Load the first image initially
  imageData = data
  loadQuestionsForCurrentImage();
  console.log('loaded')
}

async function saveData() {
  const response = await fetch('/saveData', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(imageData)
  });
  await response.json();
}

async function fetchImage(imageId) {
  const response = await fetch(`/getImage/${imageId}`, { method: 'GET' });
  const data = await response.json();
  document.getElementById('image-display').src = `data:image/jpeg;base64,${data.image}`;
}

function displayImage(file) {
  const reader = new FileReader();
  reader.onload = function(event) {
      document.getElementById('image-display').src = event.target.result;
  };
  reader.readAsDataURL(file);
}

function nextImage() {
  if (currentIndex < imageData.data.length - 1) {
    currentIndex++;
    fetchImage(currentIndex);
    loadQuestionsForCurrentImage();
  }
}

function prevImage() {
  if (currentIndex > 0) {
    currentIndex--;
    fetchImage(currentIndex);
    loadQuestionsForCurrentImage();
  }
}

function addExistingQuestion(question, answers = [], qIndex) {
  const questionsList = document.getElementById('questions-list');
  const newQuestionDiv = document.createElement('div');
  newQuestionDiv.className = 'question-div';


  const questionInput = document.createElement('input');
  questionInput.value = question || '';
  questionInput.placeholder = "Question";
  questionInput.addEventListener('keyup', (e) => updateQuestion(qIndex, e.target.value));
  questionInput.className = 'question-input';

  newQuestionDiv.appendChild(questionInput);

  answers.forEach((answer, aIndex) => {
    addExistingAnswer(newQuestionDiv, answer, qIndex, aIndex);
  });

  const addAnswerBtn = document.createElement('button');
  addAnswerBtn.innerText = "Add Answer";
  addAnswerBtn.addEventListener('click', () => addNewAnswer(newQuestionDiv, qIndex));
  newQuestionDiv.appendChild(addAnswerBtn);

  questionsList.appendChild(newQuestionDiv);
}

async function addQuestion() {
  const questionsList = document.getElementById('questions-list');
  const questionIndex = questionsList.childNodes.length;

  const newQuestion = document.createElement('div');

  const questionInput = document.createElement('input');
  questionInput.placeholder = "Question";
  questionInput.addEventListener('keyup', (e) => updateQuestion(questionIndex, e.target.value));
  const answerInput = document.createElement('input');
  answerInput.placeholder = "Answer";
  answerInput.addEventListener('keyup', (e) => updateAnswer(questionIndex, 0, e.target.value));
  newQuestion.appendChild(questionInput);
  newQuestion.appendChild(answerInput);
  questionsList.appendChild(newQuestion);
  await saveData();
}

function updateQuestion(index, question) {
  if (!imageData.data[currentIndex].questions) {
    imageData.data[currentIndex].questions = [];
  }
  if (!imageData.data[currentIndex].questions[index]) {
    imageData.data[currentIndex].questions[index] = {};
  }
  imageData.data[currentIndex].questions[index].question = question;
}


function addNewAnswer(questionDiv, qIndex) {
  const aIndex = questionDiv.childNodes.length - 2; // -2 because the last child is the Add Answer button
  console.log("Adding new answer at aIndex", aIndex);
  addExistingAnswer(questionDiv, "", qIndex, aIndex);
}

function addExistingAnswer(questionDiv, answer, qIndex, aIndex) {
  const answerInput = document.createElement('input');
  answerInput.value = answer || '';
  answerInput.placeholder = "Answer";
  answerInput.addEventListener('keyup', (e) => updateAnswer(qIndex, aIndex, e.target.value));

  questionDiv.insertBefore(answerInput, questionDiv.lastChild); // Insert before the Add Answer button
}

function updateAnswer(qIndex, aIndex, answer) {
  console.log('Updating answer', qIndex, aIndex, answer);
  if (!imageData.data[currentIndex].questions[qIndex].answers) {
    imageData.data[currentIndex].questions[qIndex].answers = [];
  }
  imageData.data[currentIndex].questions[qIndex].answers[aIndex] = answer;
}

function loadQuestionsForCurrentImage() {
  const questionsList = document.getElementById('questions-list');
  questionsList.innerHTML = '';

  const questions = imageData.data[currentIndex].questions || [];
  questions.forEach((qa, qIndex) => {
    addExistingQuestion(qa.question, qa.answers, qIndex);
  });
}