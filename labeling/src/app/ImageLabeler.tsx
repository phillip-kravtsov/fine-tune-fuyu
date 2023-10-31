"use client";
import React, { useEffect, useState } from 'react';
import styles from './ImageLabeler.module.css';

type ImageData = {
  data: {
    imageId: number,
    questions?: {
      question?: string,
      answers?: string[]
    }[]
  }[]
}
export const ImageLabeler = () => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [imageData, setImageData] = useState<ImageData>({ data: [] });
  const [currentImageSrc, setCurrentImageSrc] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    const response = await fetch('/api/getData', { method: 'POST' });
    const data = await response.json();
    setImageData(data);
    fetchImage(0);
    console.log('loaded');
  };

  const fetchImage = async (imageId: number) => {
    console.log('fetchImage');
    const response = await fetch(`/api/getImage/${imageId}`, { method: 'GET' });
    const data = await response.json();
    setCurrentImageSrc(`data:image/jpeg;base64,${data.image}`);
  };

  const saveData = async () => {
    console.log('SaveData', imageData)
    await fetch('/api/saveData', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(imageData)
    });
  };

  const nextImage = () => {
    console.log('nextImage')
    if (currentIndex < imageData.data.length - 1) {
      setCurrentIndex(currentIndex + 1);
      fetchImage(currentIndex + 1);
    }
  };

  const prevImage = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
      fetchImage(currentIndex - 1);
    }
  };
  const addQuestion = () => {
    const updatedData = { ...imageData };
    if (!updatedData.data[currentIndex].questions) {
      updatedData.data[currentIndex].questions = [];
    }
    updatedData.data[currentIndex].questions.push({ question: '', answers: [''] });
    setImageData(updatedData);
  };

  const updateQuestion = (qIndex: number, question: string) => {
    console.log('updateQuestion with index', qIndex);
    const updatedData = { ...imageData };
    updatedData.data[currentIndex].questions[qIndex].question = question;
    setImageData(updatedData);
  };

  const addAnswer = (qIndex: number) => {
    const updatedData = { ...imageData };
    if (!updatedData.data[currentIndex].questions![qIndex].answers) {
      updatedData.data[currentIndex].questions![qIndex].answers = [];
    }
    updatedData.data[currentIndex].questions[qIndex].answers.push('');
    setImageData(updatedData);
  };

  const updateAnswer = (qIndex: number, aIndex: number, answer: string) => {
    const updatedData = { ...imageData };
    if (!updatedData.data[currentIndex].questions![qIndex].answers) {
      updatedData.data[currentIndex].questions![qIndex].answers = [answer];
    } else {
      updatedData.data[currentIndex].questions[qIndex].answers[aIndex] = answer;
    }
    setImageData(updatedData);
  };

  useEffect(() => {
    fetchImage(0);  // Fetch the initial image; replace 0 with the actual initial imageId
  }, []);

  return (
    <div className={styles.mainContainer}>
      <div className={styles.topBar}>
        <button className={styles.button} onClick={saveData}>Save to File</button>
        <button className={styles.button} onClick={prevImage}>Previous</button>
        <button className={styles.button} onClick={nextImage}>Next</button>
      </div>
      <div className={styles.contentContainer}>
        <div className={styles.imageContainer}>
          <img id="image-display" src={currentImageSrc || ''} alt="Current" />
        </div>
        <div className={styles.qaContainer}>
          <button onClick={addQuestion}>Add Question</button>
          <div>
            {imageData.data[currentIndex]?.questions?.map((q, qIndex) => (
              <div key={qIndex} className={styles.questionDiv}>
                <input 
                  className={styles.questionInput}
                  value={q.question}
                  placeholder="Question"
                  onChange={(e) => updateQuestion(qIndex, e.target.value)}
                />
                {(q.answers ?? []).map((a, aIndex) => (
                  <input
                    className={styles.answerInput}
                    key={aIndex}
                    value={a}
                    placeholder="Answer"
                    onChange={(e) => updateAnswer(qIndex, aIndex, e.target.value)}
                  />
                ))}
                <button onClick={() => addAnswer(qIndex)}>Add Answer</button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}