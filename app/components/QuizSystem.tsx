'use client';

import { Brain, CheckCircle, RefreshCw, XCircle } from 'lucide-react';
import { useState } from 'react';

interface QuizQuestion {
  id: string;
  question: string;
  options: string[];
  correctAnswer: string;
  explanation: string;
  topic: 'ml_basics' | 'supervised' | 'unsupervised' | 'neural_networks' | 'cnn' | 'rnn';
}

const quizQuestions: QuizQuestion[] = [
  {
    id: 'ml1',
    topic: 'ml_basics',
    question: 'What is the primary goal of supervised learning?',
    options: [
      'To find hidden patterns in unlabeled data',
      'To learn a mapping from inputs to outputs based on labeled examples',
      'To learn through trial and error with rewards',
      'To reduce the dimensionality of data'
    ],
    correctAnswer: 'To learn a mapping from inputs to outputs based on labeled examples',
    explanation: 'Supervised learning involves training a model on a labeled dataset, where each example has input features and a known output label.'
  },
  {
    id: 'ml2',
    topic: 'ml_basics',
    question: 'Which of these is NOT a type of machine learning?',
    options: [
      'Supervised Learning',
      'Unsupervised Learning',
      'Deterministic Learning',
      'Reinforcement Learning'
    ],
    correctAnswer: 'Deterministic Learning',
    explanation: 'Deterministic learning is not a standard category of machine learning. The main types are supervised, unsupervised, and reinforcement learning.'
  },
  {
    id: 'supervised1',
    topic: 'supervised',
    question: 'Linear Regression is used for what type of problem?',
    options: [
      'Classification',
      'Regression',
      'Clustering',
      'Dimensionality Reduction'
    ],
    correctAnswer: 'Regression',
    explanation: 'Linear Regression is a fundamental algorithm for regression tasks, where the goal is to predict a continuous numerical value.'
  },
  {
    id: 'unsupervised1',
    topic: 'unsupervised',
    question: 'K-Means is an example of which type of algorithm?',
    options: [
      'Classification algorithm',
      'Regression algorithm',
      'Clustering algorithm',
      'Reinforcement learning algorithm'
    ],
    correctAnswer: 'Clustering algorithm',
    explanation: 'K-Means is a popular clustering algorithm used to partition data into K distinct, non-overlapping subgroups (clusters).'
  },
  {
    id: 'nn1',
    topic: 'neural_networks',
    question: 'What is the role of an activation function in a neural network?',
    options: [
      'To initialize the weights',
      'To calculate the loss',
      'To introduce non-linearity into the model',
      'To define the number of layers'
    ],
    correctAnswer: 'To introduce non-linearity into the model',
    explanation: 'Activation functions introduce non-linear properties to the network, allowing it to learn complex patterns and relationships in the data.'
  },
  {
    id: 'cnn1',
    topic: 'cnn',
    question: 'Which layer is primarily responsible for feature detection in CNNs?',
    options: [
      'Pooling Layer',
      'Fully Connected Layer',
      'Convolutional Layer',
      'Output Layer'
    ],
    correctAnswer: 'Convolutional Layer',
    explanation: 'Convolutional layers apply filters to input images to create feature maps, detecting patterns like edges, textures, etc.'
  },
  {
    id: 'rnn1',
    topic: 'rnn',
    question: 'What is the main advantage of LSTMs over standard RNNs?',
    options: [
      'They are faster to train',
      'They can only process image data',
      'They effectively handle long-range dependencies and mitigate vanishing gradients',
      'They require less data to train'
    ],
    correctAnswer: 'They effectively handle long-range dependencies and mitigate vanishing gradients',
    explanation: 'LSTMs (Long Short-Term Memory networks) use gating mechanisms to control the flow of information, helping them to remember information over long sequences and combat the vanishing gradient problem.'
  }
];

export default function QuizSystem() {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [score, setScore] = useState(0);
  const [showExplanation, setShowExplanation] = useState(false);
  const [quizCompleted, setQuizCompleted] = useState(false);

  const currentQuestion = quizQuestions[currentQuestionIndex];

  const handleAnswerSelection = (answer: string) => {
    if (showExplanation) return; // Don't allow changing answer after showing explanation
    setSelectedAnswer(answer);
  };

  const handleSubmitAnswer = () => {
    if (!selectedAnswer) return;

    if (selectedAnswer === currentQuestion.correctAnswer) {
      setScore(score + 1);
    }
    setShowExplanation(true);
  };

  const handleNextQuestion = () => {
    setShowExplanation(false);
    setSelectedAnswer(null);
    if (currentQuestionIndex < quizQuestions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      setQuizCompleted(true);
    }
  };

  const restartQuiz = () => {
    setCurrentQuestionIndex(0);
    setSelectedAnswer(null);
    setScore(0);
    setShowExplanation(false);
    setQuizCompleted(false);
  };

  const topicColors = {
    ml_basics: 'bg-blue-100 text-blue-800',
    supervised: 'bg-green-100 text-green-800',
    unsupervised: 'bg-purple-100 text-purple-800',
    neural_networks: 'bg-red-100 text-red-800',
    cnn: 'bg-orange-100 text-orange-800',
    rnn: 'bg-yellow-100 text-yellow-800'
  };

  if (quizCompleted) {
    return (
      <div className="max-w-2xl mx-auto px-4 py-12 text-center">
        <Brain className="w-24 h-24 mx-auto text-blue-500 mb-6" />
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Quiz Completed!</h2>
        <p className="text-xl text-gray-700 mb-6">
          Your final score is: <span className="font-bold text-blue-600">{score}</span> out of <span className="font-bold">{quizQuestions.length}</span>
        </p>
        <button
          onClick={restartQuiz}
          className="flex items-center justify-center mx-auto px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-lg shadow-md"
        >
          <RefreshCw className="w-5 h-5 mr-2" />
          Restart Quiz
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto px-4 py-12">
      <div className="text-center mb-10">
        <h2 className="text-4xl font-bold text-gray-900 mb-3">
          Test Your Knowledge 🧠
        </h2>
        <p className="text-lg text-gray-600">
          Select an answer and see how well you understand the concepts.
        </p>
      </div>

      <div className="bg-white rounded-xl shadow-2xl overflow-hidden">
        <div className={`px-6 py-4 ${topicColors[currentQuestion.topic]} flex justify-between items-center`}>
          <span className={`text-sm font-semibold`}>{currentQuestion.topic.replace('_', ' ').toUpperCase()}</span>
          <span className="text-sm font-medium">
            Question {currentQuestionIndex + 1} of {quizQuestions.length}
          </span>
        </div>

        <div className="p-6 md:p-8">
          <h3 className="text-xl font-semibold text-gray-800 mb-6 leading-relaxed">
            {currentQuestion.question}
          </h3>

          <div className="space-y-4 mb-8">
            {currentQuestion.options.map((option, index) => {
              const isSelected = selectedAnswer === option;
              const isCorrect = currentQuestion.correctAnswer === option;
              let buttonClass = 'border-gray-300 hover:border-blue-400';

              if (showExplanation) {
                if (isCorrect) {
                  buttonClass = 'border-green-500 bg-green-50 text-green-700';
                } else if (isSelected && !isCorrect) {
                  buttonClass = 'border-red-500 bg-red-50 text-red-700';
                }
              } else if (isSelected) {
                buttonClass = 'border-blue-500 bg-blue-50';
              }

              return (
                <button
                  key={index}
                  onClick={() => handleAnswerSelection(option)}
                  disabled={showExplanation}
                  className={`w-full text-left p-4 rounded-lg border-2 transition-all duration-150 flex items-center ${buttonClass} ${
                    showExplanation ? 'cursor-not-allowed' : 'cursor-pointer'
                  }`}
                >
                  <span className="flex-1 text-gray-700">{option}</span>
                  {showExplanation && isCorrect && <CheckCircle className="w-6 h-6 text-green-500 ml-3" />}
                  {showExplanation && isSelected && !isCorrect && <XCircle className="w-6 h-6 text-red-500 ml-3" />}
                </button>
              );
            })}
          </div>

          {showExplanation && (
            <div className={`p-4 rounded-lg mb-6 animate-fadeIn ${selectedAnswer === currentQuestion.correctAnswer ? 'bg-green-50 border-l-4 border-green-500' : 'bg-red-50 border-l-4 border-red-500'}`}>
              <h4 className={`text-lg font-semibold mb-2 ${selectedAnswer === currentQuestion.correctAnswer ? 'text-green-800' : 'text-red-800'}`}>
                {selectedAnswer === currentQuestion.correctAnswer ? 'Correct! 🎉' : 'Incorrect 😟'}
              </h4>
              <p className={`${selectedAnswer === currentQuestion.correctAnswer ? 'text-green-700' : 'text-red-700'} text-sm`}>
                {currentQuestion.explanation}
              </p>
            </div>
          )}

          <div className="flex justify-between items-center">
            <div className="text-lg font-semibold">
              Score: <span className="text-blue-600">{score}</span>
            </div>
            {!showExplanation ? (
              <button
                onClick={handleSubmitAnswer}
                disabled={!selectedAnswer}
                className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-base font-medium shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Submit Answer
              </button>
            ) : (
              <button
                onClick={handleNextQuestion}
                className="px-8 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-base font-medium shadow-md"
              >
                {currentQuestionIndex < quizQuestions.length - 1 ? 'Next Question →' : 'Finish Quiz'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
} 