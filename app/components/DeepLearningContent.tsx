'use client'

import { useEffect, useState } from 'react'

interface LessonProps {
  setActiveLesson: (lesson: string) => void
}

export default function DeepLearningContent() {
  const [activeLesson, setActiveLesson] = useState('neural-networks')

  const lessons = [
    { id: 'neural-networks', title: 'Neural Networks', icon: '🧠', duration: '90 min', difficulty: 'Beginner' },
    { id: 'cnn', title: 'Convolutional Networks', icon: '🖼️', duration: '120 min', difficulty: 'Intermediate' },
    { id: 'rnn', title: 'Recurrent Networks', icon: '🔄', duration: '110 min', difficulty: 'Intermediate' },
    { id: 'transformers', title: 'Transformers & Attention', icon: '⚡', duration: '140 min', difficulty: 'Advanced' },
    { id: 'advanced', title: 'Advanced Architectures', icon: '🚀', duration: '160 min', difficulty: 'Advanced' },
    { id: 'optimization', title: 'Training & Optimization', icon: '⚙️', duration: '100 min', difficulty: 'Intermediate' },
    { id: 'regularization', title: 'Regularization Techniques', icon: '🛡️', duration: '80 min', difficulty: 'Intermediate' },
    { id: 'applications', title: 'Real-World Applications', icon: '🌍', duration: '120 min', difficulty: 'All Levels' }
  ]

  const completedLessons = ['neural-networks'] // Track progress
  const currentLessonIndex = lessons.findIndex(lesson => lesson.id === activeLesson)
  const progressPercentage = ((completedLessons.length) / lessons.length) * 100

  return (
    <section className="pt-20 pb-16 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            <span className="gradient-text">Deep Learning</span> Mastery Course
          </h1>
          <p className="text-xl text-gray-600 max-w-4xl mx-auto mb-8">
            Master the fundamentals and advanced concepts of deep learning through comprehensive lessons, 
            interactive examples, and hands-on coding exercises. From neural networks to transformers and beyond.
          </p>
          <div className="flex flex-wrap justify-center gap-4 mb-8">
            <div className="bg-blue-100 text-blue-800 px-4 py-2 rounded-full text-sm font-medium">
              📚 8 Comprehensive Modules
            </div>
            <div className="bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm font-medium">
              💻 Interactive Code Examples
            </div>
            <div className="bg-purple-100 text-purple-800 px-4 py-2 rounded-full text-sm font-medium">
              🧮 Mathematical Foundations
            </div>
            <div className="bg-orange-100 text-orange-800 px-4 py-2 rounded-full text-sm font-medium">
              🎯 Practical Applications
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Enhanced Lesson Navigation */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg p-6 sticky top-24">
              <h3 className="font-semibold text-gray-900 mb-4 text-lg">Course Outline</h3>
              <nav className="space-y-3">
                {lessons.map((lesson, index) => (
                  <button
                    key={lesson.id}
                    onClick={() => setActiveLesson(lesson.id)}
                    className={`w-full text-left px-4 py-4 rounded-lg transition-all duration-200 flex flex-col space-y-2 ${
                      activeLesson === lesson.id
                        ? 'bg-purple-100 text-purple-700 border-l-4 border-purple-500 shadow-md'
                        : 'text-gray-600 hover:bg-gray-50 hover:text-purple-600 border-l-4 border-transparent'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-xl">{lesson.icon}</span>
                      <div className="flex-1">
                        <div className="font-medium text-sm">{lesson.title}</div>
                        <div className="text-xs text-gray-500 flex items-center space-x-2">
                          <span>{lesson.duration}</span>
                          <span>•</span>
                          <span className={`px-2 py-1 rounded text-xs ${
                            lesson.difficulty === 'Beginner' ? 'bg-green-100 text-green-700' :
                            lesson.difficulty === 'Intermediate' ? 'bg-yellow-100 text-yellow-700' :
                            'bg-red-100 text-red-700'
                          }`}>
                            {lesson.difficulty}
                          </span>
                        </div>
                      </div>
                    </div>
                    {completedLessons.includes(lesson.id) && (
                      <div className="flex items-center space-x-1 text-green-600 text-xs">
                        <span>✓</span>
                        <span>Completed</span>
                      </div>
                    )}
                  </button>
                ))}
              </nav>
              
              {/* Enhanced Progress */}
              <div className="mt-8 pt-6 border-t border-gray-200">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-medium text-gray-700">Overall Progress</span>
                  <span className="text-sm text-gray-500">{completedLessons.length}/{lessons.length} Complete</span>
                </div>
                <div className="progress-bar mb-4">
                  <div className="progress-fill" style={{ width: `${progressPercentage}%` }}></div>
                </div>
                <div className="text-xs text-gray-500 mb-4">
                  Estimated time remaining: {Math.max(0, lessons.length - completedLessons.length) * 2} hours
                </div>
                
                {/* Learning Path */}
                <div className="bg-gray-50 rounded-lg p-3">
                  <h4 className="font-medium text-gray-900 text-sm mb-2">Recommended Path</h4>
                  <div className="text-xs text-gray-600 space-y-1">
                    <div>1. Master fundamentals first</div>
                    <div>2. Practice with code examples</div>
                    <div>3. Complete exercises</div>
                    <div>4. Build projects</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Enhanced Main Content */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-xl shadow-lg p-8">
              {activeLesson === 'neural-networks' && <NeuralNetworksLessonWithQuiz setActiveLesson={setActiveLesson} />}
              {activeLesson === 'cnn' && <CNNLessonWithQuiz setActiveLesson={setActiveLesson} />}
              {activeLesson === 'rnn' && <RNNLessonWithQuiz setActiveLesson={setActiveLesson} />}
              {activeLesson === 'transformers' && <TransformersLessonWithQuiz setActiveLesson={setActiveLesson} />}
              {activeLesson === 'advanced' && <AdvancedLessonWithQuiz setActiveLesson={setActiveLesson} />}
              {activeLesson === 'optimization' && <OptimizationLessonWithQuiz setActiveLesson={setActiveLesson} />}
              {activeLesson === 'regularization' && <RegularizationLessonWithQuiz setActiveLesson={setActiveLesson} />}
              {activeLesson === 'applications' && <ApplicationsLessonWithQuiz setActiveLesson={setActiveLesson} />}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function NeuralNetworksLesson({ setActiveLesson }: LessonProps) {
  const [activeTab, setActiveTab] = useState('introduction')
  const [showCode, setShowCode] = useState(false)
  const [selectedExample, setSelectedExample] = useState('perceptron')

  const tabs = [
    { id: 'introduction', label: 'Introduction', icon: '📚' },
    { id: 'perceptron', label: 'Perceptron', icon: '🔵' },
    { id: 'mlp', label: 'Multi-Layer Networks', icon: '🧠' },
    { id: 'backprop', label: 'Backpropagation', icon: '🔄' },
    { id: 'activation', label: 'Activation Functions', icon: '📈' },
    { id: 'implementation', label: 'Implementation', icon: '💻' },
    { id: 'exercises', label: 'Exercises', icon: '🎯' }
  ]

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Neural Networks Fundamentals</h2>
        <p className="text-xl text-gray-600 mb-6">
          Master the building blocks of deep learning: from single neurons to complex multi-layer networks. 
          Understand the mathematical foundations, implementation details, and practical applications.
        </p>
        <div className="flex flex-wrap gap-3">
          <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">🎯 Learning Objectives</span>
          <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">📊 Mathematical Foundations</span>
          <span className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm">💻 Code Implementation</span>
        </div>
      </div>

      {/* Enhanced Tab Navigation */}
      <div className="flex flex-wrap gap-2 bg-gray-100 rounded-lg p-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-2 px-4 py-3 rounded-md text-sm font-medium transition-all duration-200 ${
              activeTab === tab.id 
                ? 'bg-white text-purple-600 shadow-sm transform scale-105' 
                : 'text-gray-600 hover:text-purple-600 hover:bg-gray-50'
            }`}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'introduction' && (
        <div className="space-y-8">
          <div className="bg-blue-50 rounded-lg p-8">
            <h3 className="text-2xl font-semibold text-blue-900 mb-6">What are Neural Networks?</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="space-y-6">
                <p className="text-blue-800 text-lg leading-relaxed">
                  Neural networks are computational models inspired by the human brain's structure and function. 
                  They consist of interconnected nodes (neurons) that process information through weighted connections.
                </p>
                
                <div className="bg-white rounded-lg p-6">
                  <h4 className="font-semibold text-blue-900 mb-4">Key Characteristics:</h4>
                  <ul className="space-y-3 text-blue-800">
                    <li className="flex items-start space-x-3">
                      <span className="text-blue-500 mt-1">•</span>
                      <div>
                        <strong>Parallel Processing:</strong> Multiple neurons work simultaneously
                      </div>
                    </li>
                    <li className="flex items-start space-x-3">
                      <span className="text-blue-500 mt-1">•</span>
                      <div>
                        <strong>Learning from Data:</strong> Adjust weights based on examples
                      </div>
                    </li>
                    <li className="flex items-start space-x-3">
                      <span className="text-blue-500 mt-1">•</span>
                      <div>
                        <strong>Non-linear Mapping:</strong> Can model complex relationships
                      </div>
                    </li>
                    <li className="flex items-start space-x-3">
                      <span className="text-blue-500 mt-1">•</span>
                      <div>
                        <strong>Fault Tolerance:</strong> Graceful degradation with damaged neurons
                      </div>
                    </li>
                  </ul>
                </div>

                <div className="bg-white rounded-lg p-6">
                  <h4 className="font-semibold text-blue-900 mb-4">Historical Timeline:</h4>
                  <div className="space-y-3 text-sm">
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                      <div><strong>1943:</strong> McCulloch-Pitts neuron model</div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                      <div><strong>1957:</strong> Perceptron by Frank Rosenblatt</div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                      <div><strong>1986:</strong> Backpropagation algorithm</div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                      <div><strong>2006:</strong> Deep learning renaissance</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <div className="bg-white rounded-lg p-6 border-2 border-dashed border-blue-300">
                  <h4 className="font-semibold text-blue-900 mb-4 text-center">Biological vs Artificial Neuron</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <h5 className="font-medium text-blue-900 mb-2">Biological Neuron</h5>
                      <ul className="space-y-1 text-blue-800">
                        <li>• Dendrites (inputs)</li>
                        <li>• Cell body (processing)</li>
                        <li>• Axon (output)</li>
                        <li>• Synapses (connections)</li>
                        <li>• Neurotransmitters</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium text-blue-900 mb-2">Artificial Neuron</h5>
                      <ul className="space-y-1 text-blue-800">
                        <li>• Input values</li>
                        <li>• Weighted sum</li>
                        <li>• Activation function</li>
                        <li>• Weight parameters</li>
                        <li>• Bias term</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg p-6">
                  <h4 className="font-semibold text-blue-900 mb-4">Applications Overview:</h4>
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="bg-blue-50 p-3 rounded">
                      <div className="font-medium text-blue-900">Computer Vision</div>
                      <div className="text-blue-700">Image classification, object detection</div>
                    </div>
                    <div className="bg-blue-50 p-3 rounded">
                      <div className="font-medium text-blue-900">Natural Language</div>
                      <div className="text-blue-700">Translation, sentiment analysis</div>
                    </div>
                    <div className="bg-blue-50 p-3 rounded">
                      <div className="font-medium text-blue-900">Speech Processing</div>
                      <div className="text-blue-700">Recognition, synthesis</div>
                    </div>
                    <div className="bg-blue-50 p-3 rounded">
                      <div className="font-medium text-blue-900">Game Playing</div>
                      <div className="text-blue-700">Chess, Go, video games</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">Learning Objectives</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <div className="bg-white rounded-lg p-4">
                <div className="text-2xl mb-2">🎯</div>
                <h4 className="font-semibold text-gray-900 mb-2">Understand Architecture</h4>
                <p className="text-gray-600 text-sm">Learn how neurons connect to form networks</p>
              </div>
              <div className="bg-white rounded-lg p-4">
                <div className="text-2xl mb-2">🧮</div>
                <h4 className="font-semibold text-gray-900 mb-2">Master Mathematics</h4>
                <p className="text-gray-600 text-sm">Grasp the mathematical foundations</p>
              </div>
              <div className="bg-white rounded-lg p-4">
                <div className="text-2xl mb-2">💻</div>
                <h4 className="font-semibold text-gray-900 mb-2">Implement Networks</h4>
                <p className="text-gray-600 text-sm">Code neural networks from scratch</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'perceptron' && (
        <div className="space-y-8">
          <div className="bg-purple-50 rounded-lg p-8">
            <h3 className="text-2xl font-semibold text-purple-900 mb-6">The Perceptron: Foundation of Neural Networks</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="space-y-6">
                <p className="text-purple-800 text-lg leading-relaxed">
                  The perceptron is the simplest form of a neural network - a single neuron that makes binary decisions. 
                  It's the building block for more complex networks and helps us understand fundamental concepts.
                </p>

                <div className="bg-white rounded-lg p-6">
                  <h4 className="font-semibold text-purple-900 mb-4">Mathematical Model:</h4>
                  <div className="bg-gray-50 rounded p-4 font-mono text-lg mb-4 text-center">
                    y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
                  </div>
                  <div className="space-y-3 text-purple-800">
                    <div><strong>x₁, x₂, ..., xₙ:</strong> Input features</div>
                    <div><strong>w₁, w₂, ..., wₙ:</strong> Weights (learnable parameters)</div>
                    <div><strong>b:</strong> Bias term (threshold adjustment)</div>
                    <div><strong>f:</strong> Activation function (step function for perceptron)</div>
                    <div><strong>y:</strong> Output (0 or 1 for binary classification)</div>
                  </div>
                </div>

                <div className="bg-white rounded-lg p-6">
                  <h4 className="font-semibold text-purple-900 mb-4">Step Function:</h4>
                  <div className="bg-gray-50 rounded p-4 font-mono text-center mb-4">
                    f(z) = {'{'}1 if z ≥ 0, 0 if z &lt; 0{'}'}
                  </div>
                  <p className="text-purple-800 text-sm">
                    The step function creates a hard decision boundary, making the perceptron suitable 
                    for linearly separable problems.
                  </p>
                </div>
              </div>

              <div className="space-y-6">
                <div className="bg-white rounded-lg p-6 border-2 border-dashed border-purple-300">
                  <h4 className="font-semibold text-purple-900 mb-4 text-center">Perceptron Visualization</h4>
                  <svg width="100%" height="300" viewBox="0 0 400 300">
                    {/* Input nodes */}
                    <circle cx="50" cy="80" r="20" className="fill-blue-200 stroke-blue-500 stroke-2" />
                    <text x="50" y="85" className="text-xs text-center" textAnchor="middle">x₁</text>
                    <text x="50" y="110" className="text-xs text-center" textAnchor="middle">Input 1</text>
                    
                    <circle cx="50" cy="160" r="20" className="fill-blue-200 stroke-blue-500 stroke-2" />
                    <text x="50" y="165" className="text-xs text-center" textAnchor="middle">x₂</text>
                    <text x="50" y="190" className="text-xs text-center" textAnchor="middle">Input 2</text>
                    
                    <circle cx="50" cy="240" r="20" className="fill-blue-200 stroke-blue-500 stroke-2" />
                    <text x="50" y="245" className="text-xs text-center" textAnchor="middle">x₃</text>
                    <text x="50" y="270" className="text-xs text-center" textAnchor="middle">Input 3</text>

                    {/* Weights */}
                    <line x1="70" y1="80" x2="180" y2="120" className="stroke-green-500 stroke-2" />
                    <text x="125" y="95" className="text-xs fill-green-600">w₁</text>
                    
                    <line x1="70" y1="160" x2="180" y2="140" className="stroke-green-500 stroke-2" />
                    <text x="125" y="155" className="text-xs fill-green-600">w₂</text>
                    
                    <line x1="70" y1="240" x2="180" y2="160" className="stroke-green-500 stroke-2" />
                    <text x="125" y="205" className="text-xs fill-green-600">w₃</text>

                    {/* Neuron */}
                    <circle cx="200" cy="140" r="30" className="fill-purple-200 stroke-purple-500 stroke-3" />
                    <text x="200" y="135" className="text-sm text-center font-bold" textAnchor="middle">Σ</text>
                    <text x="200" y="150" className="text-xs text-center" textAnchor="middle">+b</text>

                    {/* Activation */}
                    <rect x="260" y="120" width="40" height="40" className="fill-orange-200 stroke-orange-500 stroke-2" rx="5" />
                    <text x="280" y="145" className="text-sm text-center font-bold" textAnchor="middle">f</text>

                    {/* Output */}
                    <circle cx="350" cy="140" r="20" className="fill-red-200 stroke-red-500 stroke-2" />
                    <text x="350" y="145" className="text-xs text-center" textAnchor="middle">y</text>
                    <text x="350" y="170" className="text-xs text-center" textAnchor="middle">Output</text>

                    {/* Connections */}
                    <line x1="230" y1="140" x2="260" y2="140" className="stroke-gray-600 stroke-2" />
                    <line x1="300" y1="140" x2="330" y2="140" className="stroke-gray-600 stroke-2" />
                  </svg>
                </div>

                <div className="bg-white rounded-lg p-6">
                  <h4 className="font-semibold text-purple-900 mb-4">Learning Algorithm:</h4>
                  <div className="space-y-3 text-sm">
                    <div className="bg-purple-50 p-3 rounded">
                      <strong>1. Initialize:</strong> Set weights and bias to small random values
                    </div>
                    <div className="bg-purple-50 p-3 rounded">
                      <strong>2. Forward Pass:</strong> Compute output for input sample
                    </div>
                    <div className="bg-purple-50 p-3 rounded">
                      <strong>3. Compare:</strong> Check if output matches target
                    </div>
                    <div className="bg-purple-50 p-3 rounded">
                      <strong>4. Update:</strong> Adjust weights if prediction is wrong
                    </div>
                    <div className="bg-purple-50 p-3 rounded">
                      <strong>5. Repeat:</strong> Continue until convergence or max iterations
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-8 bg-white rounded-lg p-6">
              <h4 className="font-semibold text-purple-900 mb-4">Perceptron Learning Rule:</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <div className="bg-gray-50 rounded p-4 font-mono text-center mb-4">
                    wᵢ(new) = wᵢ(old) + η(target - output)xᵢ
                  </div>
                  <div className="space-y-2 text-purple-800 text-sm">
                    <div><strong>η (eta):</strong> Learning rate (typically 0.01 to 0.1)</div>
                    <div><strong>target:</strong> Desired output (0 or 1)</div>
                    <div><strong>output:</strong> Actual output from perceptron</div>
                    <div><strong>xᵢ:</strong> Input value for weight i</div>
                  </div>
                </div>
                <div>
                  <h5 className="font-medium text-purple-900 mb-3">Update Cases:</h5>
                  <div className="space-y-2 text-sm">
                    <div className="bg-green-50 p-2 rounded">
                      <strong>Correct prediction:</strong> No weight update (error = 0)
                    </div>
                    <div className="bg-red-50 p-2 rounded">
                      <strong>False positive:</strong> Decrease weights for active inputs
                    </div>
                    <div className="bg-yellow-50 p-2 rounded">
                      <strong>False negative:</strong> Increase weights for active inputs
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">Interactive Example: AND Gate</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Truth Table:</h4>
                <div className="bg-white rounded border overflow-hidden">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="p-3 text-left">x₁</th>
                        <th className="p-3 text-left">x₂</th>
                        <th className="p-3 text-left">Output</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-t">
                        <td className="p-3">0</td>
                        <td className="p-3">0</td>
                        <td className="p-3 font-bold text-red-600">0</td>
                      </tr>
                      <tr className="border-t bg-gray-50">
                        <td className="p-3">0</td>
                        <td className="p-3">1</td>
                        <td className="p-3 font-bold text-red-600">0</td>
                      </tr>
                      <tr className="border-t">
                        <td className="p-3">1</td>
                        <td className="p-3">0</td>
                        <td className="p-3 font-bold text-red-600">0</td>
                      </tr>
                      <tr className="border-t bg-gray-50">
                        <td className="p-3">1</td>
                        <td className="p-3">1</td>
                        <td className="p-3 font-bold text-green-600">1</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Learned Parameters:</h4>
                <div className="bg-white rounded p-4 space-y-3">
                  <div className="flex justify-between">
                    <span>Weight 1 (w₁):</span>
                    <span className="font-mono">0.5</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Weight 2 (w₂):</span>
                    <span className="font-mono">0.5</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Bias (b):</span>
                    <span className="font-mono">-0.7</span>
                  </div>
                  <div className="border-t pt-3">
                    <div className="text-sm text-gray-600">
                      Decision boundary: 0.5x₁ + 0.5x₂ - 0.7 = 0
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-yellow-900 mb-4">Limitations of the Perceptron</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-yellow-900 mb-3">What it CAN solve:</h4>
                <ul className="space-y-2 text-yellow-800">
                  <li className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>AND gate</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>OR gate</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>NOT gate</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <span className="text-green-500">✓</span>
                    <span>Linear classification</span>
                  </li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-yellow-900 mb-3">What it CANNOT solve:</h4>
                <ul className="space-y-2 text-yellow-800">
                  <li className="flex items-center space-x-2">
                    <span className="text-red-500">✗</span>
                    <span>XOR gate (non-linearly separable)</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <span className="text-red-500">✗</span>
                    <span>Complex pattern recognition</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <span className="text-red-500">✗</span>
                    <span>Multi-class classification</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <span className="text-red-500">✗</span>
                    <span>Non-linear relationships</span>
                  </li>
                </ul>
              </div>
            </div>
            <div className="mt-4 p-4 bg-white rounded border-l-4 border-yellow-500">
              <p className="text-yellow-800">
                <strong>Key Insight:</strong> The perceptron can only solve linearly separable problems. 
                This limitation led to the development of multi-layer perceptrons (MLPs) that can solve 
                non-linear problems by combining multiple perceptrons.
              </p>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'mlp' && (
        <div className="space-y-8">
          <div className="bg-green-50 rounded-lg p-8">
            <h3 className="text-2xl font-semibold text-green-900 mb-6">Multi-Layer Perceptron (MLP)</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="space-y-6">
                <p className="text-green-800 text-lg leading-relaxed">
                  Multi-Layer Perceptrons overcome the limitations of single perceptrons by stacking multiple 
                  layers of neurons. This architecture enables learning complex, non-linear patterns and 
                  solving problems that single perceptrons cannot handle.
                </p>

                <div className="bg-white rounded-lg p-6">
                  <h4 className="font-semibold text-green-900 mb-4">Architecture Components:</h4>
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <div className="w-6 h-6 bg-blue-500 rounded-full mt-1"></div>
                      <div>
                        <div className="font-medium text-green-900">Input Layer</div>
                        <div className="text-green-800 text-sm">Receives raw features/data</div>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-6 h-6 bg-orange-500 rounded-full mt-1"></div>
                      <div>
                        <div className="font-medium text-green-900">Hidden Layer(s)</div>
                        <div className="text-green-800 text-sm">Extract and transform features</div>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-6 h-6 bg-red-500 rounded-full mt-1"></div>
                      <div>
                        <div className="font-medium text-green-900">Output Layer</div>
                        <div className="text-green-800 text-sm">Produces final predictions</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg p-6">
                  <h4 className="font-semibold text-green-900 mb-4">Mathematical Representation:</h4>
                  <div className="space-y-3">
                    <div className="bg-gray-50 rounded p-3 font-mono text-sm">
                      <div>Layer 1: h₁ = σ(W₁x + b₁)</div>
                      <div>Layer 2: h₂ = σ(W₂h₁ + b₂)</div>
                      <div>Output: y = σ(W₃h₂ + b₃)</div>
                    </div>
                    <div className="text-green-800 text-sm">
                      Where σ is the activation function, W are weight matrices, and b are bias vectors.
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <div className="bg-white rounded-lg p-6 border-2 border-dashed border-green-300">
                  <h4 className="font-semibold text-green-900 mb-4 text-center">MLP Architecture</h4>
                  <svg width="100%" height="350" viewBox="0 0 500 350">
                    {/* Input layer */}
                    <g>
                      <circle cx="60" cy="60" r="15" className="fill-blue-200 stroke-blue-500 stroke-2" />
                      <circle cx="60" cy="120" r="15" className="fill-blue-200 stroke-blue-500 stroke-2" />
                      <circle cx="60" cy="180" r="15" className="fill-blue-200 stroke-blue-500 stroke-2" />
                      <circle cx="60" cy="240" r="15" className="fill-blue-200 stroke-blue-500 stroke-2" />
                      <text x="60" y="270" className="text-xs text-center" textAnchor="middle">Input Layer</text>
                      <text x="60" y="285" className="text-xs text-center" textAnchor="middle">(4 neurons)</text>
                    </g>
                    
                    {/* Hidden layer 1 */}
                    <g>
                      <circle cx="180" cy="40" r="15" className="fill-orange-200 stroke-orange-500 stroke-2" />
                      <circle cx="180" cy="90" r="15" className="fill-orange-200 stroke-orange-500 stroke-2" />
                      <circle cx="180" cy="140" r="15" className="fill-orange-200 stroke-orange-500 stroke-2" />
                      <circle cx="180" cy="190" r="15" className="fill-orange-200 stroke-orange-500 stroke-2" />
                      <circle cx="180" cy="240" r="15" className="fill-orange-200 stroke-orange-500 stroke-2" />
                      <circle cx="180" cy="290" r="15" className="fill-orange-200 stroke-orange-500 stroke-2" />
                      <text x="180" y="320" className="text-xs text-center" textAnchor="middle">Hidden Layer 1</text>
                      <text x="180" y="335" className="text-xs text-center" textAnchor="middle">(6 neurons)</text>
                    </g>
                    
                    {/* Hidden layer 2 */}
                    <g>
                      <circle cx="300" cy="70" r="15" className="fill-yellow-200 stroke-yellow-500 stroke-2" />
                      <circle cx="300" cy="130" r="15" className="fill-yellow-200 stroke-yellow-500 stroke-2" />
                      <circle cx="300" cy="190" r="15" className="fill-yellow-200 stroke-yellow-500 stroke-2" />
                      <circle cx="300" cy="250" r="15" className="fill-yellow-200 stroke-yellow-500 stroke-2" />
                      <text x="300" y="280" className="text-xs text-center" textAnchor="middle">Hidden Layer 2</text>
                      <text x="300" y="295" className="text-xs text-center" textAnchor="middle">(4 neurons)</text>
                    </g>
                    
                    {/* Output layer */}
                    <g>
                      <circle cx="420" cy="120" r="15" className="fill-red-200 stroke-red-500 stroke-2" />
                      <circle cx="420" cy="180" r="15" className="fill-red-200 stroke-red-500 stroke-2" />
                      <text x="420" y="210" className="text-xs text-center" textAnchor="middle">Output Layer</text>
                      <text x="420" y="225" className="text-xs text-center" textAnchor="middle">(2 neurons)</text>
                    </g>
                    
                    {/* Connections (sample) */}
                    <g className="stroke-gray-400 stroke-1 fill-none opacity-60">
                      <line x1="75" y1="60" x2="165" y2="40" />
                      <line x1="75" y1="60" x2="165" y2="90" />
                      <line x1="75" y1="120" x2="165" y2="140" />
                      <line x1="75" y1="180" x2="165" y2="190" />
                      <line x1="195" y1="90" x2="285" y2="70" />
                      <line x1="195" y1="140" x2="285" y2="130" />
                      <line x1="195" y1="190" x2="285" y2="190" />
                      <line x1="315" y1="130" x2="405" y2="120" />
                      <line x1="315" y1="190" x2="405" y2="180" />
                    </g>
                  </svg>
                </div>

                <div className="bg-white rounded-lg p-6">
                  <h4 className="font-semibold text-green-900 mb-4">Layer Specifications:</h4>
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between items-center p-2 bg-blue-50 rounded">
                      <span>Input Layer:</span>
                      <span className="font-mono">4 neurons (features)</span>
                    </div>
                    <div className="flex justify-between items-center p-2 bg-orange-50 rounded">
                      <span>Hidden Layer 1:</span>
                      <span className="font-mono">6 neurons</span>
                    </div>
                    <div className="flex justify-between items-center p-2 bg-yellow-50 rounded">
                      <span>Hidden Layer 2:</span>
                      <span className="font-mono">4 neurons</span>
                    </div>
                    <div className="flex justify-between items-center p-2 bg-red-50 rounded">
                      <span>Output Layer:</span>
                      <span className="font-mono">2 neurons (classes)</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-8 bg-white rounded-lg p-6">
              <h4 className="font-semibold text-green-900 mb-4">Universal Approximation Theorem</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <div className="bg-green-50 p-4 rounded-lg mb-4">
                    <p className="text-green-800 font-medium mb-2">Theorem Statement:</p>
                    <p className="text-green-700 text-sm">
                      A feedforward network with a single hidden layer containing a finite number of neurons 
                      can approximate any continuous function on compact subsets of Rⁿ to arbitrary accuracy.
                    </p>
                  </div>
                  <div className="space-y-2 text-green-800 text-sm">
                    <div><strong>Key Implications:</strong></div>
                    <div>• MLPs are theoretically capable of learning any pattern</div>
                    <div>• More layers can make learning more efficient</div>
                    <div>• Practical limitations exist (data, computation)</div>
                  </div>
                </div>
                <div>
                  <h5 className="font-medium text-green-900 mb-3">Practical Considerations:</h5>
                  <div className="space-y-3">
                    <div className="bg-gray-50 p-3 rounded">
                      <div className="font-medium text-gray-900">Width vs Depth</div>
                      <div className="text-gray-700 text-sm">Deeper networks often more efficient than wider ones</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded">
                      <div className="font-medium text-gray-900">Training Complexity</div>
                      <div className="text-gray-700 text-sm">More parameters require more data and computation</div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded">
                      <div className="font-medium text-gray-900">Overfitting Risk</div>
                      <div className="text-gray-700 text-sm">Complex models may memorize rather than generalize</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">Solving the XOR Problem</h3>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-gray-900 mb-3">XOR Truth Table:</h4>
                <div className="bg-white rounded border overflow-hidden mb-4">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="p-3 text-left">x₁</th>
                        <th className="p-3 text-left">x₂</th>
                        <th className="p-3 text-left">XOR Output</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-t">
                        <td className="p-3">0</td>
                        <td className="p-3">0</td>
                        <td className="p-3 font-bold text-red-600">0</td>
                      </tr>
                      <tr className="border-t bg-gray-50">
                        <td className="p-3">0</td>
                        <td className="p-3">1</td>
                        <td className="p-3 font-bold text-green-600">1</td>
                      </tr>
                      <tr className="border-t">
                        <td className="p-3">1</td>
                        <td className="p-3">0</td>
                        <td className="p-3 font-bold text-green-600">1</td>
                      </tr>
                      <tr className="border-t bg-gray-50">
                        <td className="p-3">1</td>
                        <td className="p-3">1</td>
                        <td className="p-3 font-bold text-red-600">0</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <div className="bg-yellow-50 p-3 rounded border-l-4 border-yellow-500">
                  <p className="text-yellow-800 text-sm">
                    <strong>Why single perceptron fails:</strong> XOR is not linearly separable. 
                    No single line can separate the positive and negative examples.
                  </p>
                </div>
              </div>
              <div>
                <h4 className="font-medium text-gray-900 mb-3">MLP Solution:</h4>
                <div className="bg-white rounded p-4 space-y-4">
                  <div>
                    <div className="font-medium text-gray-900 mb-2">Architecture: 2-2-1</div>
                    <div className="text-gray-700 text-sm">2 inputs → 2 hidden neurons → 1 output</div>
                  </div>
                  <div className="bg-gray-50 p-3 rounded font-mono text-sm">
                    <div>Hidden Layer:</div>
                    <div>h₁ = σ(w₁₁x₁ + w₁₂x₂ + b₁)</div>
                    <div>h₂ = σ(w₂₁x₁ + w₂₂x₂ + b₂)</div>
                    <div className="mt-2">Output Layer:</div>
                    <div>y = σ(w₃₁h₁ + w₃₂h₂ + b₃)</div>
                  </div>
                  <div>
                    <div className="font-medium text-gray-900 mb-2">Learned Weights (example):</div>
                    <div className="text-sm space-y-1">
                      <div>w₁₁=20, w₁₂=20, b₁=-10 (OR-like)</div>
                      <div>w₂₁=20, w₂₂=20, b₂=-30 (AND-like)</div>
                      <div>w₃₁=20, w₃₂=-20, b₃=-10 (difference)</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'backprop' && (
        <div className="space-y-8">
          <div className="bg-blue-50 rounded-lg p-8">
            <h3 className="text-2xl font-semibold text-blue-900 mb-6">Backpropagation Algorithm</h3>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="space-y-6">
                <p className="text-blue-800 text-lg leading-relaxed">
                  Backpropagation is the cornerstone algorithm that enables neural networks to learn. 
                  It efficiently computes gradients by propagating errors backward through the network, 
                  allowing us to update weights to minimize the loss function.
                </p>

                <div className="bg-white rounded-lg p-6">
                  <h4 className="font-semibold text-blue-900 mb-4">Algorithm Overview:</h4>
                  <div className="space-y-4">
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">1</div>
                      <div>
                        <div className="font-medium text-blue-900">Forward Pass</div>
                        <div className="text-blue-800 text-sm">Compute outputs layer by layer</div>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">2</div>
                      <div>
                        <div className="font-medium text-blue-900">Compute Loss</div>
                        <div className="text-blue-800 text-sm">Calculate error between prediction and target</div>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">3</div>
                      <div>
                        <div className="font-medium text-blue-900">Backward Pass</div>
                        <div className="text-blue-800 text-sm">Propagate gradients backward</div>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-sm font-bold">4</div>
                      <div>
                        <div className="font-medium text-blue-900">Update Weights</div>
                        <div className="text-blue-800 text-sm">Adjust parameters using gradients</div>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-lg p-6">
                  <h4 className="font-semibold text-blue-900 mb-4">Mathematical Foundation:</h4>
                  <div className="space-y-3">
                    <div className="bg-gray-50 rounded p-3">
                      <div className="font-medium text-blue-900 mb-2">Chain Rule:</div>
                      <div className="font-mono text-sm">∂L/∂w = ∂L/∂y × ∂y/∂z × ∂z/∂w</div>
                    </div>
                    <div className="text-blue-800 text-sm">
                      The chain rule allows us to decompose complex derivatives into simpler parts, 
                      enabling efficient gradient computation through the network.
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <div className="bg-white rounded-lg p-6 border-2 border-dashed border-blue-300">
                  <h4 className="font-semibold text-blue-900 mb-4 text-center">Gradient Flow</h4>
                  <svg width="100%" height="300" viewBox="0 0 400 300">
                    {/* Network structure */}
                    <circle cx="50" cy="100" r="20" className="fill-green-200 stroke-green-500 stroke-2" />
                    <text x="50" y="105" className="text-xs text-center" textAnchor="middle">Input</text>
                    
                    <circle cx="150" cy="80" r="20" className="fill-orange-200 stroke-orange-500 stroke-2" />
                    <circle cx="150" cy="120" r="20" className="fill-orange-200 stroke-orange-500 stroke-2" />
                    <text x="150" y="150" className="text-xs text-center" textAnchor="middle">Hidden</text>
                    
                    <circle cx="250" cy="100" r="20" className="fill-red-200 stroke-red-500 stroke-2" />
                    <text x="250" y="130" className="text-xs text-center" textAnchor="middle">Output</text>
                    
                    <circle cx="350" cy="100" r="20" className="fill-purple-200 stroke-purple-500 stroke-2" />
                    <text x="350" y="130" className="text-xs text-center" textAnchor="middle">Loss</text>

                    {/* Forward arrows */}
                    <line x1="70" y1="100" x2="130" y2="85" className="stroke-green-600 stroke-2" markerEnd="url(#arrowhead)" />
                    <line x1="70" y1="100" x2="130" y2="115" className="stroke-green-600 stroke-2" markerEnd="url(#arrowhead)" />
                    <line x1="170" y1="85" x2="230" y2="95" className="stroke-orange-600 stroke-2" markerEnd="url(#arrowhead)" />
                    <line x1="170" y1="115" x2="230" y2="105" className="stroke-orange-600 stroke-2" markerEnd="url(#arrowhead)" />
                    <line x1="270" y1="100" x2="330" y2="100" className="stroke-red-600 stroke-2" markerEnd="url(#arrowhead)" />

                    {/* Backward arrows (gradients) */}
                    <line x1="330" y1="110" x2="270" y2="110" className="stroke-purple-600 stroke-2 stroke-dasharray-5-5" markerEnd="url(#arrowhead)" />
                    <line x1="230" y1="110" x2="170" y2="125" className="stroke-purple-600 stroke-2 stroke-dasharray-5-5" markerEnd="url(#arrowhead)" />
                    <line x1="230" y1="110" x2="170" y2="95" className="stroke-purple-600 stroke-2 stroke-dasharray-5-5" markerEnd="url(#arrowhead)" />
                    <line x1="130" y1="125" x2="70" y2="110" className="stroke-purple-600 stroke-2 stroke-dasharray-5-5" markerEnd="url(#arrowhead)" />
                    <line x1="130" y1="95" x2="70" y2="110" className="stroke-purple-600 stroke-2 stroke-dasharray-5-5" markerEnd="url(#arrowhead)" />

                    {/* Labels */}
                    <text x="200" y="50" className="text-xs text-green-600">Forward Pass</text>
                    <text x="200" y="180" className="text-xs text-purple-600">Backward Pass (Gradients)</text>

                    <defs>
                      <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" className="fill-current" />
                      </marker>
                    </defs>
                  </svg>
                </div>

                <div className="bg-white rounded-lg p-6">
                  <h4 className="font-semibold text-blue-900 mb-4">Gradient Computation:</h4>
                  <div className="space-y-3 text-sm">
                    <div className="bg-blue-50 p-3 rounded">
                      <div className="font-medium text-blue-900">Output Layer:</div>
                      <div className="font-mono">∂L/∂w_out = δ_out × h</div>
                    </div>
                    <div className="bg-blue-50 p-3 rounded">
                      <div className="font-medium text-blue-900">Hidden Layer:</div>
                      <div className="font-mono">∂L/∂w_hid = δ_hid × x</div>
                    </div>
                    <div className="bg-blue-50 p-3 rounded">
                      <div className="font-medium text-blue-900">Error Terms:</div>
                      <div className="font-mono">δ = ∂L/∂z × σ'(z)</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-8 bg-white rounded-lg p-6">
              <h4 className="font-semibold text-blue-900 mb-6">Detailed Backpropagation Steps</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <h5 className="font-medium text-blue-900 mb-4">Forward Pass Equations:</h5>
                  <div className="space-y-3 text-sm">
                    <div className="bg-gray-50 p-3 rounded font-mono">
                      <div>z₁ = W₁x + b₁</div>
                      <div>h₁ = σ(z₁)</div>
                      <div>z₂ = W₂h₁ + b₂</div>
                      <div>ŷ = σ(z₂)</div>
                      <div>L = ½(y - ŷ)²</div>
                    </div>
                    <div className="text-blue-800">
                      <strong>Where:</strong> z = pre-activation, h = hidden activation, 
                      σ = activation function, L = loss
                    </div>
                  </div>
                </div>
                <div>
                  <h5 className="font-medium text-blue-900 mb-4">Backward Pass Equations:</h5>
                  <div className="space-y-3 text-sm">
                    <div className="bg-gray-50 p-3 rounded font-mono">
                      <div>∂L/∂ŷ = -(y - ŷ)</div>
                      <div>∂L/∂z₂ = ∂L/∂ŷ × σ'(z₂)</div>
                      <div>∂L/∂W₂ = ∂L/∂z₂ × h₁ᵀ</div>
                      <div>∂L/∂h₁ = W₂ᵀ × ∂L/∂z₂</div>
                      <div>∂L/∂z₁ = ∂L/∂h₁ × σ'(z₁)</div>
                      <div>∂L/∂W₁ = ∂L/∂z₁ × xᵀ</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-yellow-50 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-yellow-900 mb-4">Common Challenges & Solutions</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-yellow-900 mb-3">Vanishing Gradients:</h4>
                <div className="space-y-3">
                  <div className="bg-white p-3 rounded">
                    <div className="font-medium text-red-700">Problem:</div>
                    <div className="text-red-600 text-sm">Gradients become very small in deep networks</div>
                  </div>
                  <div className="bg-white p-3 rounded">
                    <div className="font-medium text-green-700">Solutions:</div>
                    <ul className="text-green-600 text-sm space-y-1">
                      <li>• Better activation functions (ReLU)</li>
                      <li>• Proper weight initialization</li>
                      <li>• Batch normalization</li>
                      <li>• Residual connections</li>
                    </ul>
                  </div>
                </div>
              </div>
              <div>
                <h4 className="font-medium text-yellow-900 mb-3">Exploding Gradients:</h4>
                <div className="space-y-3">
                  <div className="bg-white p-3 rounded">
                    <div className="font-medium text-red-700">Problem:</div>
                    <div className="text-red-600 text-sm">Gradients become very large, causing instability</div>
                  </div>
                  <div className="bg-white p-3 rounded">
                    <div className="font-medium text-green-700">Solutions:</div>
                    <ul className="text-green-600 text-sm space-y-1">
                      <li>• Gradient clipping</li>
                      <li>• Lower learning rates</li>
                      <li>• Better weight initialization</li>
                      <li>• Batch normalization</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'activation' && (
        <div className="space-y-8">
          <div className="bg-purple-50 rounded-lg p-8">
            <h3 className="text-2xl font-semibold text-purple-900 mb-6">Activation Functions</h3>
            
            <p className="text-purple-800 text-lg leading-relaxed mb-8">
              Activation functions introduce non-linearity into neural networks, enabling them to learn 
              complex patterns. The choice of activation function significantly impacts training dynamics 
              and model performance.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Sigmoid */}
              <div className="bg-white rounded-lg p-6 border-2 border-purple-200">
                <h4 className="font-semibold text-purple-900 mb-4">Sigmoid (Logistic)</h4>
                <div className="bg-gray-50 rounded p-3 font-mono text-sm mb-4 text-center">
                  σ(x) = 1/(1 + e⁻ˣ)
                </div>
                <div className="h-32 bg-gray-100 rounded mb-4 flex items-center justify-center">
                  <svg width="120" height="80" viewBox="0 0 120 80">
                    <path d="M 10 70 Q 60 40 110 10" stroke="#8B5CF6" strokeWidth="2" fill="none" />
                    <line x1="10" y1="40" x2="110" y2="40" stroke="#E5E7EB" strokeWidth="1" />
                    <line x1="60" y1="10" x2="60" y2="70" stroke="#E5E7EB" strokeWidth="1" />
                  </svg>
                </div>
                <div className="space-y-2 text-sm">
                  <div><strong>Range:</strong> (0, 1)</div>
                  <div><strong>Pros:</strong> Smooth, differentiable</div>
                  <div><strong>Cons:</strong> Vanishing gradients, not zero-centered</div>
                  <div><strong>Use:</strong> Binary classification output</div>
                </div>
              </div>

              {/* Tanh */}
              <div className="bg-white rounded-lg p-6 border-2 border-purple-200">
                <h4 className="font-semibold text-purple-900 mb-4">Hyperbolic Tangent</h4>
                <div className="bg-gray-50 rounded p-3 font-mono text-sm mb-4 text-center">
                  tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)
                </div>
                <div className="h-32 bg-gray-100 rounded mb-4 flex items-center justify-center">
                  <svg width="120" height="80" viewBox="0 0 120 80">
                    <path d="M 10 60 Q 30 50 50 40 Q 70 30 90 20 Q 100 15 110 10" stroke="#8B5CF6" strokeWidth="2" fill="none" />
                    <line x1="10" y1="40" x2="110" y2="40" stroke="#E5E7EB" strokeWidth="1" />
                    <line x1="60" y1="10" x2="60" y2="70" stroke="#E5E7EB" strokeWidth="1" />
                  </svg>
                </div>
                <div className="space-y-2 text-sm">
                  <div><strong>Range:</strong> (-1, 1)</div>
                  <div><strong>Pros:</strong> Zero-centered, smooth</div>
                  <div><strong>Cons:</strong> Still has vanishing gradients</div>
                  <div><strong>Use:</strong> Hidden layers (better than sigmoid)</div>
                </div>
              </div>

              {/* ReLU */}
              <div className="bg-white rounded-lg p-6 border-2 border-green-200">
                <h4 className="font-semibold text-green-900 mb-4">ReLU (Rectified Linear)</h4>
                <div className="bg-gray-50 rounded p-3 font-mono text-sm mb-4 text-center">
                  ReLU(x) = max(0, x)
                </div>
                <div className="h-32 bg-gray-100 rounded mb-4 flex items-center justify-center">
                  <svg width="120" height="80" viewBox="0 0 120 80">
                    <path d="M 10 70 L 60 70 L 110 20" stroke="#10B981" strokeWidth="2" fill="none" />
                    <line x1="10" y1="40" x2="110" y2="40" stroke="#E5E7EB" strokeWidth="1" />
                    <line x1="60" y1="10" x2="60" y2="70" stroke="#E5E7EB" strokeWidth="1" />
                  </svg>
                </div>
                <div className="space-y-2 text-sm">
                  <div><strong>Range:</strong> [0, ∞)</div>
                  <div><strong>Pros:</strong> No vanishing gradients, computationally efficient</div>
                  <div><strong>Cons:</strong> Dead neurons, not zero-centered</div>
                  <div><strong>Use:</strong> Most popular for hidden layers</div>
                </div>
              </div>

              {/* Leaky ReLU */}
              <div className="bg-white rounded-lg p-6 border-2 border-blue-200">
                <h4 className="font-semibold text-blue-900 mb-4">Leaky ReLU</h4>
                <div className="bg-gray-50 rounded p-3 font-mono text-sm mb-4 text-center">
                  f(x) = max(αx, x), α ≈ 0.01
                </div>
                <div className="h-32 bg-gray-100 rounded mb-4 flex items-center justify-center">
                  <svg width="120" height="80" viewBox="0 0 120 80">
                    <path d="M 10 75 L 60 65 L 110 15" stroke="#3B82F6" strokeWidth="2" fill="none" />
                    <line x1="10" y1="40" x2="110" y2="40" stroke="#E5E7EB" strokeWidth="1" />
                    <line x1="60" y1="10" x2="60" y2="70" stroke="#E5E7EB" strokeWidth="1" />
                  </svg>
                </div>
                <div className="space-y-2 text-sm">
                  <div><strong>Range:</strong> (-∞, ∞)</div>
                  <div><strong>Pros:</strong> Fixes dead ReLU problem</div>
                  <div><strong>Cons:</strong> Still not zero-centered</div>
                  <div><strong>Use:</strong> Alternative to ReLU</div>
                </div>
              </div>

              {/* ELU */}
              <div className="bg-white rounded-lg p-6 border-2 border-orange-200">
                <h4 className="font-semibold text-orange-900 mb-4">ELU (Exponential Linear)</h4>
                <div className="bg-gray-50 rounded p-3 font-mono text-sm mb-4 text-center">
                  f(x) = {'{'}x if x &gt; 0, α(eˣ - 1) if x ≤ 0{'}'}
                </div>
                <div className="h-32 bg-gray-100 rounded mb-4 flex items-center justify-center">
                  <svg width="120" height="80" viewBox="0 0 120 80">
                    <path d="M 10 70 Q 40 65 60 60 L 110 10" stroke="#F97316" strokeWidth="2" fill="none" />
                    <line x1="10" y1="40" x2="110" y2="40" stroke="#E5E7EB" strokeWidth="1" />
                    <line x1="60" y1="10" x2="60" y2="70" stroke="#E5E7EB" strokeWidth="1" />
                  </svg>
                </div>
                <div className="space-y-2 text-sm">
                  <div><strong>Range:</strong> (-α, ∞)</div>
                  <div><strong>Pros:</strong> Smooth, zero-centered mean</div>
                  <div><strong>Cons:</strong> Computationally expensive</div>
                  <div><strong>Use:</strong> When smooth activation needed</div>
                </div>
              </div>

              {/* Swish */}
              <div className="bg-white rounded-lg p-6 border-2 border-red-200">
                <h4 className="font-semibold text-red-900 mb-4">Swish (SiLU)</h4>
                <div className="bg-gray-50 rounded p-3 font-mono text-sm mb-4 text-center">
                  f(x) = x × σ(x) = x/(1 + e⁻ˣ)
                </div>
                <div className="h-32 bg-gray-100 rounded mb-4 flex items-center justify-center">
                  <svg width="120" height="80" viewBox="0 0 120 80">
                    <path d="M 10 65 Q 30 60 50 50 Q 70 35 90 25 Q 100 20 110 15" stroke="#EF4444" strokeWidth="2" fill="none" />
                    <line x1="10" y1="40" x2="110" y2="40" stroke="#E5E7EB" strokeWidth="1" />
                    <line x1="60" y1="10" x2="60" y2="70" stroke="#E5E7EB" strokeWidth="1" />
                  </svg>
                </div>
                <div className="space-y-2 text-sm">
                  <div><strong>Range:</strong> (-∞, ∞)</div>
                  <div><strong>Pros:</strong> Smooth, self-gated</div>
                  <div><strong>Cons:</strong> More complex computation</div>
                  <div><strong>Use:</strong> Modern architectures (Transformers)</div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">Activation Function Comparison</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="bg-white">
                    <th className="p-3 text-left">Function</th>
                    <th className="p-3 text-left">Range</th>
                    <th className="p-3 text-left">Derivative</th>
                    <th className="p-3 text-left">Vanishing Gradient</th>
                    <th className="p-3 text-left">Zero-Centered</th>
                    <th className="p-3 text-left">Computational Cost</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t">
                    <td className="p-3 font-medium">Sigmoid</td>
                    <td className="p-3">(0, 1)</td>
                    <td className="p-3">σ(x)(1-σ(x))</td>
                    <td className="p-3 text-red-600">Yes</td>
                    <td className="p-3 text-red-600">No</td>
                    <td className="p-3 text-yellow-600">Medium</td>
                  </tr>
                  <tr className="border-t bg-gray-50">
                    <td className="p-3 font-medium">Tanh</td>
                    <td className="p-3">(-1, 1)</td>
                    <td className="p-3">1 - tanh²(x)</td>
                    <td className="p-3 text-red-600">Yes</td>
                    <td className="p-3 text-green-600">Yes</td>
                    <td className="p-3 text-yellow-600">Medium</td>
                  </tr>
                  <tr className="border-t">
                    <td className="p-3 font-medium">ReLU</td>
                    <td className="p-3">[0, ∞)</td>
                    <td className="p-3">{'{'}1 if x&gt;0, 0 if x≤0{'}'}</td>
                    <td className="p-3 text-green-600">No</td>
                    <td className="p-3 text-red-600">No</td>
                    <td className="p-3 text-green-600">Low</td>
                  </tr>
                  <tr className="border-t bg-gray-50">
                    <td className="p-3 font-medium">Leaky ReLU</td>
                    <td className="p-3">(-∞, ∞)</td>
                    <td className="p-3">{'{'}1 if x&gt;0, α if x≤0{'}'}</td>
                    <td className="p-3 text-green-600">No</td>
                    <td className="p-3 text-red-600">No</td>
                    <td className="p-3 text-green-600">Low</td>
                  </tr>
                  <tr className="border-t">
                    <td className="p-3 font-medium">Swish</td>
                    <td className="p-3">(-∞, ∞)</td>
                    <td className="p-3">Complex</td>
                    <td className="p-3 text-green-600">No</td>
                    <td className="p-3 text-yellow-600">Partial</td>
                    <td className="p-3 text-red-600">High</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-blue-50 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-blue-900 mb-4">Choosing the Right Activation Function</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-blue-900 mb-3">Hidden Layers:</h4>
                <div className="space-y-3">
                  <div className="bg-white p-3 rounded">
                    <div className="font-medium text-green-700">✓ ReLU (Default choice)</div>
                    <div className="text-gray-600 text-sm">Fast, effective, widely used</div>
                  </div>
                  <div className="bg-white p-3 rounded">
                    <div className="font-medium text-blue-700">✓ Leaky ReLU</div>
                    <div className="text-gray-600 text-sm">When dead neurons are a problem</div>
                  </div>
                  <div className="bg-white p-3 rounded">
                    <div className="font-medium text-purple-700">✓ Swish/GELU</div>
                    <div className="text-gray-600 text-sm">For modern architectures</div>
                  </div>
                </div>
              </div>
              <div>
                <h4 className="font-medium text-blue-900 mb-3">Output Layers:</h4>
                <div className="space-y-3">
                  <div className="bg-white p-3 rounded">
                    <div className="font-medium text-green-700">✓ Sigmoid</div>
                    <div className="text-gray-600 text-sm">Binary classification</div>
                  </div>
                  <div className="bg-white p-3 rounded">
                    <div className="font-medium text-blue-700">✓ Softmax</div>
                    <div className="text-gray-600 text-sm">Multi-class classification</div>
                  </div>
                  <div className="bg-white p-3 rounded">
                    <div className="font-medium text-purple-700">✓ Linear (None)</div>
                    <div className="text-gray-600 text-sm">Regression tasks</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'implementation' && (
        <div className="space-y-8">
          <div className="bg-green-50 rounded-lg p-8">
            <h3 className="text-2xl font-semibold text-green-900 mb-6">Neural Network Implementation</h3>
            
            <div className="flex items-center justify-between mb-6">
              <p className="text-green-800 text-lg">
                Let's implement a neural network from scratch to understand the underlying mechanics.
              </p>
              <button
                onClick={() => setShowCode(!showCode)}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
              >
                {showCode ? 'Hide Code' : 'Show Code'}
              </button>
            </div>

            {showCode && (
              <div className="space-y-6">
                <div className="bg-gray-900 rounded-lg p-6 overflow-x-auto">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-white font-medium">Python Implementation</h4>
                    <div className="flex space-x-2">
                      <button
                        onClick={() => setSelectedExample('perceptron')}
                        className={`px-3 py-1 rounded text-sm ${
                          selectedExample === 'perceptron' 
                            ? 'bg-green-600 text-white' 
                            : 'bg-gray-700 text-gray-300'
                        }`}
                      >
                        Perceptron
                      </button>
                      <button
                        onClick={() => setSelectedExample('mlp')}
                        className={`px-3 py-1 rounded text-sm ${
                          selectedExample === 'mlp' 
                            ? 'bg-green-600 text-white' 
                            : 'bg-gray-700 text-gray-300'
                        }`}
                      >
                        MLP
                      </button>
                      <button
                        onClick={() => setSelectedExample('numpy')}
                        className={`px-3 py-1 rounded text-sm ${
                          selectedExample === 'numpy' 
                            ? 'bg-green-600 text-white' 
                            : 'bg-gray-700 text-gray-300'
                        }`}
                      >
                        NumPy
                      </button>
                    </div>
                  </div>
                  
                  {selectedExample === 'perceptron' && (
                    <pre className="text-green-400 text-sm overflow-x-auto">
{`import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        # Training loop
        for _ in range(self.max_iter):
            for i in range(X.shape[0]):
                # Forward pass
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.activation(linear_output)
                
                # Update weights if prediction is wrong
                if prediction != y[i]:
                    error = y[i] - prediction
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
    
    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.array([self.activation(x) for x in linear_output])

# Example usage
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # AND gate

perceptron = Perceptron()
perceptron.fit(X, y)
predictions = perceptron.predict(X)
print(f"Predictions: {predictions}")
print(f"Weights: {perceptron.weights}")
print(f"Bias: {perceptron.bias}")`}
                    </pre>
                  )}

                  {selectedExample === 'mlp' && (
                    <pre className="text-green-400 text-sm overflow-x-auto">
{`import numpy as np

class MLP:
    def __init__(self, layers, learning_rate=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.1
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.activations = [X]
        
        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Calculate output layer error
        error = output - y
        deltas = [error * self.sigmoid_derivative(output)]
        
        # Backpropagate error
        for i in range(len(self.weights) - 2, -1, -1):
            error = deltas[-1].dot(self.weights[i + 1].T)
            delta = error * self.sigmoid_derivative(self.activations[i + 1])
            deltas.append(delta)
        
        deltas.reverse()
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.activations[i].T.dot(deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        return self.forward(X)

# Example: XOR problem
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create MLP: 2 inputs -> 4 hidden -> 1 output
mlp = MLP([2, 4, 1])
mlp.train(X, y, 1000)

predictions = mlp.predict(X)
print("\\nXOR Predictions:")
for i in range(len(X)):
    print(f"{X[i]} -> {predictions[i][0]:.3f}")`}
                    </pre>
                  )}

                  {selectedExample === 'numpy' && (
                    <pre className="text-green-400 text-sm overflow-x-auto">
{`import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, architecture):
        self.architecture = architecture
        self.parameters = {}
        self.cache = {}
        self.gradients = {}
        
        # Initialize parameters
        for i in range(1, len(architecture)):
            self.parameters[f'W{i}'] = np.random.randn(
                architecture[i-1], architecture[i]
            ) * np.sqrt(2.0 / architecture[i-1])  # He initialization
            self.parameters[f'b{i}'] = np.zeros((1, architecture[i]))
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        return (Z > 0).astype(float)
    
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-np.clip(Z, -250, 250)))
    
    def forward_propagation(self, X):
        self.cache['A0'] = X
        
        for i in range(1, len(self.architecture)):
            Z = np.dot(self.cache[f'A{i-1}'], self.parameters[f'W{i}']) + self.parameters[f'b{i}']
            
            if i == len(self.architecture) - 1:  # Output layer
                A = self.sigmoid(Z)
            else:  # Hidden layers
                A = self.relu(Z)
            
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = A
        
        return self.cache[f'A{len(self.architecture)-1}']
    
    def compute_cost(self, Y_hat, Y):
        m = Y.shape[0]
        cost = -np.mean(Y * np.log(Y_hat + 1e-8) + (1 - Y) * np.log(1 - Y_hat + 1e-8))
        return cost
    
    def backward_propagation(self, X, Y):
        m = X.shape[0]
        L = len(self.architecture) - 1
        
        # Output layer
        dZ = self.cache[f'A{L}'] - Y
        self.gradients[f'dW{L}'] = np.dot(self.cache[f'A{L-1}'].T, dZ) / m
        self.gradients[f'db{L}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Hidden layers
        for i in range(L-1, 0, -1):
            dA = np.dot(dZ, self.parameters[f'W{i+1}'].T)
            dZ = dA * self.relu_derivative(self.cache[f'Z{i}'])
            self.gradients[f'dW{i}'] = np.dot(self.cache[f'A{i-1}'].T, dZ) / m
            self.gradients[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
    
    def update_parameters(self, learning_rate):
        for i in range(1, len(self.architecture)):
            self.parameters[f'W{i}'] -= learning_rate * self.gradients[f'dW{i}']
            self.parameters[f'b{i}'] -= learning_rate * self.gradients[f'db{i}']
    
    def train(self, X, Y, epochs, learning_rate=0.01):
        costs = []
        
        for epoch in range(epochs):
            # Forward propagation
            Y_hat = self.forward_propagation(X)
            
            # Compute cost
            cost = self.compute_cost(Y_hat, Y)
            costs.append(cost)
            
            # Backward propagation
            self.backward_propagation(X, Y)
            
            # Update parameters
            self.update_parameters(learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {cost:.4f}")
        
        return costs
    
    def predict(self, X):
        Y_hat = self.forward_propagation(X)
        return (Y_hat > 0.5).astype(int)

# Example usage
np.random.seed(42)

# Generate spiral dataset
def generate_spiral_data(n_points=100):
    X = np.zeros((n_points*2, 2))
    y = np.zeros(n_points*2)
    
    for j in range(2):
        ix = range(n_points*j, n_points*(j+1))
        r = np.linspace(0.0, 1, n_points)
        t = np.linspace(j*4, (j+1)*4, n_points) + np.random.randn(n_points)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    
    return X, y.reshape(-1, 1)

X, y = generate_spiral_data(100)

# Create and train network
nn = NeuralNetwork([2, 10, 10, 1])
costs = nn.train(X, y, epochs=2000, learning_rate=0.1)

# Make predictions
predictions = nn.predict(X)
accuracy = np.mean(predictions == y)
print(f"\\nAccuracy: {accuracy:.2%}")`}
                    </pre>
                  )}
                </div>

                <div className="bg-white rounded-lg p-6">
                  <h4 className="font-semibold text-green-900 mb-4">Key Implementation Details:</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h5 className="font-medium text-green-900 mb-3">Weight Initialization:</h5>
                      <ul className="space-y-2 text-green-800 text-sm">
                        <li>• <strong>Xavier/Glorot:</strong> For sigmoid/tanh</li>
                        <li>• <strong>He initialization:</strong> For ReLU</li>
                        <li>• <strong>Random small values:</strong> Break symmetry</li>
                        <li>• <strong>Zero bias:</strong> Usually safe to start with</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium text-green-900 mb-3">Training Tips:</h5>
                      <ul className="space-y-2 text-green-800 text-sm">
                        <li>• <strong>Learning rate:</strong> Start with 0.01-0.1</li>
                        <li>• <strong>Batch processing:</strong> More stable gradients</li>
                        <li>• <strong>Monitoring:</strong> Track loss and accuracy</li>
                        <li>• <strong>Early stopping:</strong> Prevent overfitting</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="bg-blue-50 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-blue-900 mb-4">Framework Implementations</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-blue-900 mb-3">TensorFlow/Keras</h4>
                <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=100)`}
                </pre>
              </div>
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-blue-900 mb-3">PyTorch</h4>
                <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
{`import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)`}
                </pre>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'exercises' && (
        <div className="space-y-8">
          <div className="bg-orange-50 rounded-lg p-8">
            <h3 className="text-2xl font-semibold text-orange-900 mb-6">Hands-On Exercises</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* Exercise 1 */}
              <div className="bg-white rounded-lg p-6 border-2 border-orange-200">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">1</div>
                  <h4 className="font-semibold text-orange-900">Perceptron Logic Gates</h4>
                </div>
                <p className="text-orange-800 text-sm mb-4">
                  Implement perceptrons for AND, OR, and NOT gates. Understand why XOR cannot be solved.
                </p>
                <div className="space-y-3">
                  <div className="bg-orange-50 p-3 rounded">
                    <div className="font-medium text-orange-900 text-sm">Tasks:</div>
                    <ul className="text-orange-800 text-xs space-y-1 mt-1">
                      <li>• Code perceptron from scratch</li>
                      <li>• Train on logic gate data</li>
                      <li>• Visualize decision boundaries</li>
                      <li>• Analyze weight values</li>
                    </ul>
                  </div>
                  <div className="text-xs text-orange-700">
                    <strong>Difficulty:</strong> Beginner | <strong>Time:</strong> 30 min
                  </div>
                </div>
              </div>

              {/* Exercise 2 */}
              <div className="bg-white rounded-lg p-6 border-2 border-orange-200">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">2</div>
                  <h4 className="font-semibold text-orange-900">XOR with MLP</h4>
                </div>
                <p className="text-orange-800 text-sm mb-4">
                  Build a multi-layer perceptron to solve the XOR problem that single perceptrons cannot handle.
                </p>
                <div className="space-y-3">
                  <div className="bg-orange-50 p-3 rounded">
                    <div className="font-medium text-orange-900 text-sm">Tasks:</div>
                    <ul className="text-orange-800 text-xs space-y-1 mt-1">
                      <li>• Design 2-2-1 architecture</li>
                      <li>• Implement backpropagation</li>
                      <li>• Train until convergence</li>
                      <li>• Visualize hidden representations</li>
                    </ul>
                  </div>
                  <div className="text-xs text-orange-700">
                    <strong>Difficulty:</strong> Intermediate | <strong>Time:</strong> 45 min
                  </div>
                </div>
              </div>

              {/* Exercise 3 */}
              <div className="bg-white rounded-lg p-6 border-2 border-orange-200">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">3</div>
                  <h4 className="font-semibold text-orange-900">Activation Comparison</h4>
                </div>
                <p className="text-orange-800 text-sm mb-4">
                  Compare different activation functions on the same dataset and analyze their impact.
                </p>
                <div className="space-y-3">
                  <div className="bg-orange-50 p-3 rounded">
                    <div className="font-medium text-orange-900 text-sm">Tasks:</div>
                    <ul className="text-orange-800 text-xs space-y-1 mt-1">
                      <li>• Test sigmoid, tanh, ReLU</li>
                      <li>• Compare training speed</li>
                      <li>• Analyze gradient flow</li>
                      <li>• Plot learning curves</li>
                    </ul>
                  </div>
                  <div className="text-xs text-orange-700">
                    <strong>Difficulty:</strong> Intermediate | <strong>Time:</strong> 60 min
                  </div>
                </div>
              </div>

              {/* Exercise 4 */}
              <div className="bg-white rounded-lg p-6 border-2 border-orange-200">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">4</div>
                  <h4 className="font-semibold text-orange-900">Iris Classification</h4>
                </div>
                <p className="text-orange-800 text-sm mb-4">
                  Build a neural network to classify iris flowers using the classic dataset.
                </p>
                <div className="space-y-3">
                  <div className="bg-orange-50 p-3 rounded">
                    <div className="font-medium text-orange-900 text-sm">Tasks:</div>
                    <ul className="text-orange-800 text-xs space-y-1 mt-1">
                      <li>• Load and preprocess data</li>
                      <li>• Design multi-class network</li>
                      <li>• Implement softmax output</li>
                      <li>• Evaluate performance</li>
                    </ul>
                  </div>
                  <div className="text-xs text-orange-700">
                    <strong>Difficulty:</strong> Intermediate | <strong>Time:</strong> 75 min
                  </div>
                </div>
              </div>

              {/* Exercise 5 */}
              <div className="bg-white rounded-lg p-6 border-2 border-orange-200">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">5</div>
                  <h4 className="font-semibold text-orange-900">Gradient Checking</h4>
                </div>
                <p className="text-orange-800 text-sm mb-4">
                  Implement numerical gradient checking to verify your backpropagation implementation.
                </p>
                <div className="space-y-3">
                  <div className="bg-orange-50 p-3 rounded">
                    <div className="font-medium text-orange-900 text-sm">Tasks:</div>
                    <ul className="text-orange-800 text-xs space-y-1 mt-1">
                      <li>• Implement finite differences</li>
                      <li>• Compare with analytical gradients</li>
                      <li>• Debug backpropagation errors</li>
                      <li>• Understand numerical precision</li>
                    </ul>
                  </div>
                  <div className="text-xs text-orange-700">
                    <strong>Difficulty:</strong> Advanced | <strong>Time:</strong> 90 min
                  </div>
                </div>
              </div>

              {/* Exercise 6 */}
              <div className="bg-white rounded-lg p-6 border-2 border-orange-200">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-8 h-8 bg-orange-500 text-white rounded-full flex items-center justify-center font-bold">6</div>
                  <h4 className="font-semibold text-orange-900">Hyperparameter Tuning</h4>
                </div>
                <p className="text-orange-800 text-sm mb-4">
                  Systematically explore the effect of different hyperparameters on network performance.
                </p>
                <div className="space-y-3">
                  <div className="bg-orange-50 p-3 rounded">
                    <div className="font-medium text-orange-900 text-sm">Tasks:</div>
                    <ul className="text-orange-800 text-xs space-y-1 mt-1">
                      <li>• Vary learning rates</li>
                      <li>• Test different architectures</li>
                      <li>• Compare initialization methods</li>
                      <li>• Create performance heatmaps</li>
                    </ul>
                  </div>
                  <div className="text-xs text-orange-700">
                    <strong>Difficulty:</strong> Advanced | <strong>Time:</strong> 120 min
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">Project Ideas</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 mb-3">🎯 Beginner Projects</h4>
                <ul className="space-y-2 text-gray-700 text-sm">
                  <li>• House price prediction (regression)</li>
                  <li>• Binary sentiment classification</li>
                  <li>• Handwritten digit recognition (MNIST)</li>
                  <li>• Wine quality classification</li>
                </ul>
              </div>
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-gray-900 mb-3">🚀 Advanced Projects</h4>
                <ul className="space-y-2 text-gray-700 text-sm">
                  <li>• Multi-output regression</li>
                  <li>• Imbalanced dataset classification</li>
                  <li>• Time series forecasting</li>
                  <li>• Custom loss function design</li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-blue-50 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-blue-900 mb-4">Assessment Checklist</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-blue-900 mb-3">Theoretical Understanding:</h4>
                <div className="space-y-2">
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded" />
                    <span className="text-blue-800 text-sm">Understand perceptron limitations</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded" />
                    <span className="text-blue-800 text-sm">Explain backpropagation algorithm</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded" />
                    <span className="text-blue-800 text-sm">Compare activation functions</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded" />
                    <span className="text-blue-800 text-sm">Understand universal approximation</span>
                  </label>
                </div>
              </div>
              <div>
                <h4 className="font-medium text-blue-900 mb-3">Practical Skills:</h4>
                <div className="space-y-2">
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded" />
                    <span className="text-blue-800 text-sm">Implement neural network from scratch</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded" />
                    <span className="text-blue-800 text-sm">Debug training problems</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded" />
                    <span className="text-blue-800 text-sm">Choose appropriate architectures</span>
                  </label>
                  <label className="flex items-center space-x-2">
                    <input type="checkbox" className="rounded" />
                    <span className="text-blue-800 text-sm">Evaluate model performance</span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Navigation */}
      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <div className="text-sm text-gray-500">
          Lesson 1 of 8 • Neural Networks Fundamentals
        </div>
        <button 
          onClick={() => setActiveLesson('cnn')}
          className="px-8 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all duration-200 transform hover:scale-105 shadow-lg"
        >
          Next: Convolutional Networks →
        </button>
      </div>
    </div>
  )
}

function CNNLesson({ setActiveLesson }: LessonProps) {
  const [activeTab, setActiveTab] = useState('introduction')

  const tabs = [
    { id: 'introduction', label: 'Introduction', icon: '📚' },
    { id: 'convolution', label: 'Convolution Operation', icon: '🔍' },
    { id: 'pooling', label: 'Pooling Layers', icon: '📊' },
    { id: 'architectures', label: 'CNN Architectures', icon: '🏗️' },
    { id: 'implementation', label: 'Implementation', icon: '💻' },
    { id: 'applications', label: 'Applications', icon: '🎯' }
  ]

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-blue-50 to-green-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Convolutional Neural Networks</h2>
        <p className="text-xl text-gray-600 mb-6">
          Master CNNs for computer vision tasks. Learn convolution operations, pooling, and modern architectures 
          that revolutionized image processing and pattern recognition.
        </p>
      </div>

      <div className="flex flex-wrap gap-2 bg-gray-100 rounded-lg p-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-2 px-4 py-3 rounded-md text-sm font-medium transition-all duration-200 ${
              activeTab === tab.id 
                ? 'bg-white text-blue-600 shadow-sm transform scale-105' 
                : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
            }`}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {activeTab === 'introduction' && (
        <div className="space-y-6">
          <div className="bg-blue-50 rounded-lg p-6">
            <h3 className="text-2xl font-semibold text-blue-900 mb-4">What are CNNs?</h3>
            <p className="text-blue-800 text-lg mb-4">
              Convolutional Neural Networks are specialized neural networks designed for processing grid-like data 
              such as images. They use convolution operations to detect local features and patterns.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-blue-900 mb-3">Key Advantages:</h4>
                <ul className="space-y-2 text-blue-800 text-sm">
                  <li>• Translation invariance</li>
                  <li>• Parameter sharing</li>
                  <li>• Hierarchical feature learning</li>
                  <li>• Reduced overfitting</li>
                </ul>
              </div>
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-blue-900 mb-3">Applications:</h4>
                <ul className="space-y-2 text-blue-800 text-sm">
                  <li>• Image classification</li>
                  <li>• Object detection</li>
                  <li>• Medical imaging</li>
                  <li>• Autonomous vehicles</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('neural-networks')}
          className="px-6 py-2 text-gray-600 hover:text-blue-600 transition-colors"
        >
          ← Previous: Neural Networks
        </button>
        <div className="text-sm text-gray-500">Lesson 2 of 8 • Convolutional Networks</div>
        <button 
          onClick={() => setActiveLesson('rnn')}
          className="px-8 py-3 bg-gradient-to-r from-blue-600 to-green-600 text-white rounded-lg hover:from-blue-700 hover:to-green-700 transition-all duration-200"
        >
          Next: Recurrent Networks →
        </button>
      </div>
    </div>
  )
}

function RNNLesson({ setActiveLesson }: LessonProps) {
  const [activeTab, setActiveTab] = useState('introduction')

  const tabs = [
    { id: 'introduction', label: 'Introduction', icon: '📚' },
    { id: 'vanilla-rnn', label: 'Vanilla RNN', icon: '🔄' },
    { id: 'lstm', label: 'LSTM', icon: '🧠' },
    { id: 'gru', label: 'GRU', icon: '⚡' },
    { id: 'applications', label: 'Applications', icon: '🎯' }
  ]

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-green-50 to-purple-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Recurrent Neural Networks</h2>
        <p className="text-xl text-gray-600 mb-6">
          Learn RNNs for sequential data processing. Understand LSTM, GRU, and how to handle 
          time series, natural language, and other sequential patterns.
        </p>
      </div>

      <div className="flex flex-wrap gap-2 bg-gray-100 rounded-lg p-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-2 px-4 py-3 rounded-md text-sm font-medium transition-all duration-200 ${
              activeTab === tab.id 
                ? 'bg-white text-green-600 shadow-sm transform scale-105' 
                : 'text-gray-600 hover:text-green-600 hover:bg-gray-50'
            }`}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {activeTab === 'introduction' && (
        <div className="space-y-6">
          <div className="bg-green-50 rounded-lg p-6">
            <h3 className="text-2xl font-semibold text-green-900 mb-4">Sequential Data Processing</h3>
            <p className="text-green-800 text-lg mb-4">
              RNNs are designed to work with sequential data where the order matters. They maintain 
              internal memory to process sequences of varying lengths.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-green-900 mb-3">RNN Types:</h4>
                <ul className="space-y-2 text-green-800 text-sm">
                  <li>• One-to-many (image captioning)</li>
                  <li>• Many-to-one (sentiment analysis)</li>
                  <li>• Many-to-many (translation)</li>
                  <li>• Sequence-to-sequence</li>
                </ul>
              </div>
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-green-900 mb-3">Challenges:</h4>
                <ul className="space-y-2 text-green-800 text-sm">
                  <li>• Vanishing gradients</li>
                  <li>• Long-term dependencies</li>
                  <li>• Computational complexity</li>
                  <li>• Training instability</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('cnn')}
          className="px-6 py-2 text-gray-600 hover:text-green-600 transition-colors"
        >
          ← Previous: CNNs
        </button>
        <div className="text-sm text-gray-500">Lesson 3 of 8 • Recurrent Networks</div>
        <button 
          onClick={() => setActiveLesson('transformers')}
          className="px-8 py-3 bg-gradient-to-r from-green-600 to-purple-600 text-white rounded-lg hover:from-green-700 hover:to-purple-700 transition-all duration-200"
        >
          Next: Transformers →
        </button>
      </div>
    </div>
  )
}

function TransformersLesson({ setActiveLesson }: LessonProps) {
  const [activeTab, setActiveTab] = useState('introduction')

  const tabs = [
    { id: 'introduction', label: 'Introduction', icon: '📚' },
    { id: 'attention', label: 'Attention Mechanism', icon: '👁️' },
    { id: 'architecture', label: 'Transformer Architecture', icon: '🏗️' },
    { id: 'variants', label: 'Variants', icon: '🔄' },
    { id: 'applications', label: 'Applications', icon: '🎯' }
  ]

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Transformers & Attention</h2>
        <p className="text-xl text-gray-600 mb-6">
          Explore the revolutionary Transformer architecture that powers modern NLP. Learn attention 
          mechanisms, self-attention, and how transformers changed AI forever.
        </p>
      </div>

      <div className="flex flex-wrap gap-2 bg-gray-100 rounded-lg p-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-2 px-4 py-3 rounded-md text-sm font-medium transition-all duration-200 ${
              activeTab === tab.id 
                ? 'bg-white text-purple-600 shadow-sm transform scale-105' 
                : 'text-gray-600 hover:text-purple-600 hover:bg-gray-50'
            }`}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {activeTab === 'introduction' && (
        <div className="space-y-6">
          <div className="bg-purple-50 rounded-lg p-6">
            <h3 className="text-2xl font-semibold text-purple-900 mb-4">The Attention Revolution</h3>
            <p className="text-purple-800 text-lg mb-4">
              Transformers introduced the "Attention is All You Need" paradigm, replacing recurrence 
              with self-attention mechanisms for better parallelization and long-range dependencies.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-purple-900 mb-3">Key Innovations:</h4>
                <ul className="space-y-2 text-purple-800 text-sm">
                  <li>• Self-attention mechanism</li>
                  <li>• Parallel processing</li>
                  <li>• Positional encoding</li>
                  <li>• Multi-head attention</li>
                </ul>
              </div>
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-purple-900 mb-3">Impact:</h4>
                <ul className="space-y-2 text-purple-800 text-sm">
                  <li>• GPT series (language models)</li>
                  <li>• BERT (bidirectional encoding)</li>
                  <li>• Vision Transformers (ViT)</li>
                  <li>• Multimodal models</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('rnn')}
          className="px-6 py-2 text-gray-600 hover:text-purple-600 transition-colors"
        >
          ← Previous: RNNs
        </button>
        <div className="text-sm text-gray-500">Lesson 4 of 8 • Transformers & Attention</div>
        <button 
          onClick={() => setActiveLesson('advanced')}
          className="px-8 py-3 bg-gradient-to-r from-purple-600 to-red-600 text-white rounded-lg hover:from-purple-700 hover:to-red-700 transition-all duration-200"
        >
          Next: Advanced Architectures →
        </button>
      </div>
    </div>
  )
}

function AdvancedLesson({ setActiveLesson }: LessonProps) {
  const [activeTab, setActiveTab] = useState('introduction')

  const tabs = [
    { id: 'introduction', label: 'Introduction', icon: '📚' },
    { id: 'gans', label: 'GANs', icon: '🎭' },
    { id: 'autoencoders', label: 'Autoencoders', icon: '🔄' },
    { id: 'diffusion', label: 'Diffusion Models', icon: '🌊' },
    { id: 'nerf', label: 'Neural Fields', icon: '🌐' }
  ]

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-red-50 to-orange-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Advanced Architectures</h2>
        <p className="text-xl text-gray-600 mb-6">
          Explore cutting-edge neural network architectures including GANs, autoencoders, 
          diffusion models, and neural fields that push the boundaries of AI.
        </p>
      </div>

      <div className="flex flex-wrap gap-2 bg-gray-100 rounded-lg p-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-2 px-4 py-3 rounded-md text-sm font-medium transition-all duration-200 ${
              activeTab === tab.id 
                ? 'bg-white text-red-600 shadow-sm transform scale-105' 
                : 'text-gray-600 hover:text-red-600 hover:bg-gray-50'
            }`}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {activeTab === 'introduction' && (
        <div className="space-y-6">
          <div className="bg-red-50 rounded-lg p-6">
            <h3 className="text-2xl font-semibold text-red-900 mb-4">Next-Generation Architectures</h3>
            <p className="text-red-800 text-lg mb-4">
              Advanced architectures tackle complex problems like generation, representation learning, 
              and novel data modalities with innovative approaches and mathematical frameworks.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-red-900 mb-3">Generative Models:</h4>
                <ul className="space-y-2 text-red-800 text-sm">
                  <li>• GANs (adversarial training)</li>
                  <li>• VAEs (variational inference)</li>
                  <li>• Diffusion models</li>
                  <li>• Flow-based models</li>
                </ul>
              </div>
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-red-900 mb-3">Specialized Architectures:</h4>
                <ul className="space-y-2 text-red-800 text-sm">
                  <li>• Graph neural networks</li>
                  <li>• Neural ODEs</li>
                  <li>• Memory networks</li>
                  <li>• Capsule networks</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('transformers')}
          className="px-6 py-2 text-gray-600 hover:text-red-600 transition-colors"
        >
          ← Previous: Transformers
        </button>
        <div className="text-sm text-gray-500">Lesson 5 of 8 • Advanced Architectures</div>
        <button 
          onClick={() => setActiveLesson('optimization')}
          className="px-8 py-3 bg-gradient-to-r from-red-600 to-yellow-600 text-white rounded-lg hover:from-red-700 hover:to-yellow-700 transition-all duration-200"
        >
          Next: Training & Optimization →
        </button>
      </div>
    </div>
  )
}

function OptimizationLesson({ setActiveLesson }: LessonProps) {
  const [activeTab, setActiveTab] = useState('introduction')

  const tabs = [
    { id: 'introduction', label: 'Introduction', icon: '📚' },
    { id: 'optimizers', label: 'Optimizers', icon: '⚙️' },
    { id: 'learning-rate', label: 'Learning Rate', icon: '📈' },
    { id: 'batch-norm', label: 'Normalization', icon: '⚖️' },
    { id: 'techniques', label: 'Advanced Techniques', icon: '🎯' }
  ]

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-yellow-50 to-green-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Training & Optimization</h2>
        <p className="text-xl text-gray-600 mb-6">
          Master the art of training neural networks efficiently. Learn about optimizers, 
          learning rate schedules, normalization techniques, and advanced training strategies.
        </p>
      </div>

      <div className="flex flex-wrap gap-2 bg-gray-100 rounded-lg p-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-2 px-4 py-3 rounded-md text-sm font-medium transition-all duration-200 ${
              activeTab === tab.id 
                ? 'bg-white text-yellow-600 shadow-sm transform scale-105' 
                : 'text-gray-600 hover:text-yellow-600 hover:bg-gray-50'
            }`}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {activeTab === 'introduction' && (
        <div className="space-y-6">
          <div className="bg-yellow-50 rounded-lg p-6">
            <h3 className="text-2xl font-semibold text-yellow-900 mb-4">Optimization Fundamentals</h3>
            <p className="text-yellow-800 text-lg mb-4">
              Training neural networks is an optimization problem. We need to find the best parameters 
              that minimize the loss function while ensuring good generalization.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-yellow-900 mb-3">Key Challenges:</h4>
                <ul className="space-y-2 text-yellow-800 text-sm">
                  <li>• Non-convex loss landscapes</li>
                  <li>• Saddle points and local minima</li>
                  <li>• Vanishing/exploding gradients</li>
                  <li>• Slow convergence</li>
                </ul>
              </div>
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-yellow-900 mb-3">Solutions:</h4>
                <ul className="space-y-2 text-yellow-800 text-sm">
                  <li>• Adaptive optimizers</li>
                  <li>• Learning rate scheduling</li>
                  <li>• Normalization techniques</li>
                  <li>• Regularization methods</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('advanced')}
          className="px-6 py-2 text-gray-600 hover:text-yellow-600 transition-colors"
        >
          ← Previous: Advanced Architectures
        </button>
        <div className="text-sm text-gray-500">Lesson 6 of 8 • Training & Optimization</div>
        <button 
          onClick={() => setActiveLesson('regularization')}
          className="px-8 py-3 bg-gradient-to-r from-yellow-600 to-blue-600 text-white rounded-lg hover:from-yellow-700 hover:to-blue-700 transition-all duration-200"
        >
          Next: Regularization →
        </button>
      </div>
    </div>
  )
}

function RegularizationLesson({ setActiveLesson }: LessonProps) {
  const [activeTab, setActiveTab] = useState('introduction')

  const tabs = [
    { id: 'introduction', label: 'Introduction', icon: '📚' },
    { id: 'dropout', label: 'Dropout', icon: '🎲' },
    { id: 'weight-decay', label: 'Weight Decay', icon: '⚖️' },
    { id: 'data-aug', label: 'Data Augmentation', icon: '🔄' },
    { id: 'early-stopping', label: 'Early Stopping', icon: '⏹️' }
  ]

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Regularization Techniques</h2>
        <p className="text-xl text-gray-600 mb-6">
          Learn how to prevent overfitting and improve generalization. Master dropout, weight decay, 
          data augmentation, and other techniques that make models robust.
        </p>
      </div>

      <div className="flex flex-wrap gap-2 bg-gray-100 rounded-lg p-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-2 px-4 py-3 rounded-md text-sm font-medium transition-all duration-200 ${
              activeTab === tab.id 
                ? 'bg-white text-blue-600 shadow-sm transform scale-105' 
                : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
            }`}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {activeTab === 'introduction' && (
        <div className="space-y-6">
          <div className="bg-blue-50 rounded-lg p-6">
            <h3 className="text-2xl font-semibold text-blue-900 mb-4">Fighting Overfitting</h3>
            <p className="text-blue-800 text-lg mb-4">
              Regularization techniques help models generalize better by preventing them from 
              memorizing training data and encouraging simpler, more robust representations.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-blue-900 mb-3">Signs of Overfitting:</h4>
                <ul className="space-y-2 text-blue-800 text-sm">
                  <li>• High training, low validation accuracy</li>
                  <li>• Large gap between train/val loss</li>
                  <li>• Poor performance on new data</li>
                  <li>• Model memorizes noise</li>
                </ul>
              </div>
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-blue-900 mb-3">Regularization Types:</h4>
                <ul className="space-y-2 text-blue-800 text-sm">
                  <li>• Explicit (L1, L2 penalties)</li>
                  <li>• Implicit (dropout, noise)</li>
                  <li>• Data-based (augmentation)</li>
                  <li>• Architecture-based (batch norm)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('optimization')}
          className="px-6 py-2 text-gray-600 hover:text-blue-600 transition-colors"
        >
          ← Previous: Optimization
        </button>
        <div className="text-sm text-gray-500">Lesson 7 of 8 • Regularization Techniques</div>
        <button 
          onClick={() => setActiveLesson('applications')}
          className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200"
        >
          Next: Applications →
        </button>
      </div>
    </div>
  )
}

function ApplicationsLesson({ setActiveLesson }: LessonProps) {
  const [activeTab, setActiveTab] = useState('introduction')

  const tabs = [
    { id: 'introduction', label: 'Introduction', icon: '📚' },
    { id: 'computer-vision', label: 'Computer Vision', icon: '👁️' },
    { id: 'nlp', label: 'Natural Language', icon: '💬' },
    { id: 'healthcare', label: 'Healthcare', icon: '🏥' },
    { id: 'industry', label: 'Industry 4.0', icon: '🏭' },
    { id: 'future', label: 'Future Trends', icon: '🚀' }
  ]

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Real-World Applications</h2>
        <p className="text-xl text-gray-600 mb-6">
          Explore how deep learning transforms industries and solves real-world problems. 
          From healthcare to autonomous systems, see AI in action across domains.
        </p>
      </div>

      <div className="flex flex-wrap gap-2 bg-gray-100 rounded-lg p-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-2 px-4 py-3 rounded-md text-sm font-medium transition-all duration-200 ${
              activeTab === tab.id 
                ? 'bg-white text-purple-600 shadow-sm transform scale-105' 
                : 'text-gray-600 hover:text-purple-600 hover:bg-gray-50'
            }`}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {activeTab === 'introduction' && (
        <div className="space-y-6">
          <div className="bg-purple-50 rounded-lg p-6">
            <h3 className="text-2xl font-semibold text-purple-900 mb-4">AI Transforming the World</h3>
            <p className="text-purple-800 text-lg mb-4">
              Deep learning has moved from research labs to production systems, revolutionizing 
              how we work, communicate, and solve complex problems across every industry.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-purple-900 mb-3">🎯 Current Impact:</h4>
                <ul className="space-y-2 text-purple-800 text-sm">
                  <li>• Autonomous vehicles</li>
                  <li>• Medical diagnosis</li>
                  <li>• Language translation</li>
                  <li>• Content recommendation</li>
                </ul>
              </div>
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-purple-900 mb-3">🚀 Emerging Areas:</h4>
                <ul className="space-y-2 text-purple-800 text-sm">
                  <li>• Drug discovery</li>
                  <li>• Climate modeling</li>
                  <li>• Robotics</li>
                  <li>• Creative AI</li>
                </ul>
              </div>
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-purple-900 mb-3">🔮 Future Potential:</h4>
                <ul className="space-y-2 text-purple-800 text-sm">
                  <li>• AGI development</li>
                  <li>• Scientific discovery</li>
                  <li>• Space exploration</li>
                  <li>• Quantum computing</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('regularization')}
          className="px-6 py-2 text-gray-600 hover:text-purple-600 transition-colors"
        >
          ← Previous: Regularization
        </button>
        <div className="text-sm text-gray-500">Lesson 8 of 8 • Real-World Applications</div>
        <div className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg">
          🎉 Course Complete!
        </div>
      </div>
    </div>
  )
}

// Quiz System Components
interface QuizQuestion {
  id: string
  type: 'multiple-choice' | 'true-false' | 'fill-blank' | 'code-completion' | 'drag-drop'
  question: string
  options?: string[]
  correctAnswer: string | string[]
  explanation: string
  difficulty: 'easy' | 'medium' | 'hard'
  points: number
  code?: string
  blanks?: string[]
}

interface QuizProps {
  lessonId: string
  questions: QuizQuestion[]
  onComplete: (score: number, totalPoints: number) => void
}

function Quiz({ lessonId, questions, onComplete }: QuizProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [answers, setAnswers] = useState<Record<string, string | string[]>>({})
  const [showExplanation, setShowExplanation] = useState(false)
  const [quizCompleted, setQuizCompleted] = useState(false)
  const [score, setScore] = useState(0)
  const [timeSpent, setTimeSpent] = useState(0)

  const question = questions[currentQuestion]
  const totalQuestions = questions.length
  const progress = ((currentQuestion + 1) / totalQuestions) * 100

  useEffect(() => {
    const timer = setInterval(() => {
      setTimeSpent(prev => prev + 1)
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  const handleAnswer = (answer: string | string[]) => {
    setAnswers(prev => ({ ...prev, [question.id]: answer }))
  }

  const checkAnswer = () => {
    const userAnswer = answers[question.id]
    const correct = Array.isArray(question.correctAnswer) 
      ? JSON.stringify(userAnswer) === JSON.stringify(question.correctAnswer)
      : userAnswer === question.correctAnswer
    
    if (correct) {
      setScore(prev => prev + question.points)
    }
    setShowExplanation(true)
  }

  const nextQuestion = () => {
    if (currentQuestion < totalQuestions - 1) {
      setCurrentQuestion(prev => prev + 1)
      setShowExplanation(false)
    } else {
      const totalPoints = questions.reduce((sum, q) => sum + q.points, 0)
      setQuizCompleted(true)
      onComplete(score, totalPoints)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  if (quizCompleted) {
    const totalPoints = questions.reduce((sum, q) => sum + q.points, 0)
    const percentage = Math.round((score / totalPoints) * 100)
    
    return (
      <div className="bg-white rounded-xl shadow-lg p-8">
        <div className="text-center">
          <div className="text-6xl mb-4">
            {percentage >= 90 ? '🏆' : percentage >= 80 ? '🥇' : percentage >= 70 ? '🥈' : percentage >= 60 ? '🥉' : '📚'}
          </div>
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Quiz Complete!</h2>
          <div className="text-xl text-gray-600 mb-6">
            You scored {score} out of {totalPoints} points ({percentage}%)
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-blue-600">{percentage}%</div>
              <div className="text-blue-800">Final Score</div>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-green-600">{formatTime(timeSpent)}</div>
              <div className="text-green-800">Time Taken</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-4">
              <div className="text-2xl font-bold text-purple-600">{totalQuestions}</div>
              <div className="text-purple-800">Questions</div>
            </div>
          </div>

          <div className="mb-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-3">Performance Analysis</h3>
            <div className="text-left bg-gray-50 rounded-lg p-4">
              {percentage >= 90 && (
                <div className="text-green-700">
                  🎉 Excellent! You have mastered this topic. You're ready for advanced concepts.
                </div>
              )}
              {percentage >= 80 && percentage < 90 && (
                <div className="text-blue-700">
                  👍 Great job! You have a solid understanding. Review a few concepts and you'll be perfect.
                </div>
              )}
              {percentage >= 70 && percentage < 80 && (
                <div className="text-yellow-700">
                  📖 Good work! You understand the basics. Spend more time on the challenging topics.
                </div>
              )}
              {percentage >= 60 && percentage < 70 && (
                <div className="text-orange-700">
                  🔄 You're getting there! Review the lesson materials and try the quiz again.
                </div>
              )}
              {percentage < 60 && (
                <div className="text-red-700">
                  📚 Keep studying! Go back to the lesson content and practice more before retaking.
                </div>
              )}
            </div>
          </div>

          <div className="flex justify-center space-x-4">
            <button 
              onClick={() => window.location.reload()}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Retake Quiz
            </button>
            <button 
              onClick={() => setQuizCompleted(false)}
              className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
            >
              Review Answers
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-8">
      {/* Quiz Header */}
      <div className="mb-8">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-2xl font-bold text-gray-900">
            {lessonId.charAt(0).toUpperCase() + lessonId.slice(1)} Quiz
          </h2>
          <div className="text-sm text-gray-500">
            Time: {formatTime(timeSpent)}
          </div>
        </div>
        
        <div className="flex justify-between items-center mb-4">
          <span className="text-sm text-gray-600">
            Question {currentQuestion + 1} of {totalQuestions}
          </span>
          <span className="text-sm text-gray-600">
            {question.points} points • {question.difficulty}
          </span>
        </div>
        
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          ></div>
        </div>
      </div>

      {/* Question Content */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold text-gray-900 mb-6">{question.question}</h3>
        
        {question.code && (
          <div className="bg-gray-900 rounded-lg p-4 mb-6">
            <pre className="text-green-400 text-sm overflow-x-auto">
              <code>{question.code}</code>
            </pre>
          </div>
        )}

        {/* Multiple Choice */}
        {question.type === 'multiple-choice' && (
          <div className="space-y-3">
            {question.options?.map((option, index) => (
              <button
                key={index}
                onClick={() => handleAnswer(option)}
                className={`w-full text-left p-4 rounded-lg border-2 transition-all duration-200 ${
                  answers[question.id] === option
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <span className="font-medium text-gray-700">
                  {String.fromCharCode(65 + index)}. {option}
                </span>
              </button>
            ))}
          </div>
        )}

        {/* True/False */}
        {question.type === 'true-false' && (
          <div className="flex space-x-4">
            <button
              onClick={() => handleAnswer('true')}
              className={`flex-1 p-4 rounded-lg border-2 transition-all duration-200 ${
                answers[question.id] === 'true'
                  ? 'border-green-500 bg-green-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <span className="font-medium text-gray-700">✓ True</span>
            </button>
            <button
              onClick={() => handleAnswer('false')}
              className={`flex-1 p-4 rounded-lg border-2 transition-all duration-200 ${
                answers[question.id] === 'false'
                  ? 'border-red-500 bg-red-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <span className="font-medium text-gray-700">✗ False</span>
            </button>
          </div>
        )}

        {/* Fill in the Blank */}
        {question.type === 'fill-blank' && (
          <div className="space-y-4">
            {question.blanks?.map((blank, index) => (
              <div key={index}>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Fill in blank {index + 1}:
                </label>
                <input
                  type="text"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Enter your answer..."
                  onChange={(e) => {
                    const currentAnswers = Array.isArray(answers[question.id]) 
                      ? [...(answers[question.id] as string[])] 
                      : new Array(question.blanks?.length).fill('')
                    currentAnswers[index] = e.target.value
                    handleAnswer(currentAnswers)
                  }}
                />
              </div>
            ))}
          </div>
        )}

        {/* Code Completion */}
        {question.type === 'code-completion' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Complete the code:
            </label>
            <textarea
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm"
              rows={6}
              placeholder="Write your code here..."
              onChange={(e) => handleAnswer(e.target.value)}
            />
          </div>
        )}
      </div>

      {/* Explanation */}
      {showExplanation && (
        <div className="mb-8 p-6 bg-gray-50 rounded-lg">
          <h4 className="font-semibold text-gray-900 mb-3">
            {answers[question.id] === question.correctAnswer || 
             JSON.stringify(answers[question.id]) === JSON.stringify(question.correctAnswer)
              ? '✅ Correct!' 
              : '❌ Incorrect'}
          </h4>
          <p className="text-gray-700 mb-3">{question.explanation}</p>
          <div className="text-sm text-gray-600">
            <strong>Correct answer:</strong> {
              Array.isArray(question.correctAnswer) 
                ? question.correctAnswer.join(', ')
                : question.correctAnswer
            }
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex justify-between">
        <button
          onClick={() => setCurrentQuestion(Math.max(0, currentQuestion - 1))}
          disabled={currentQuestion === 0}
          className="px-6 py-3 text-gray-600 hover:text-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          ← Previous
        </button>
        
        <div className="space-x-4">
          {!showExplanation ? (
            <button
              onClick={checkAnswer}
              disabled={!answers[question.id]}
              className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Submit Answer
            </button>
          ) : (
            <button
              onClick={nextQuestion}
              className="px-8 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
            >
              {currentQuestion < totalQuestions - 1 ? 'Next Question →' : 'Finish Quiz'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

// Quiz Data for Each Lesson
const neuralNetworksQuiz: QuizQuestion[] = [
  {
    id: 'nn-1',
    type: 'multiple-choice',
    question: 'What is the main limitation of a single perceptron?',
    options: [
      'It can only solve linearly separable problems',
      'It requires too much computational power',
      'It cannot learn from data',
      'It only works with binary inputs'
    ],
    correctAnswer: 'It can only solve linearly separable problems',
    explanation: 'A single perceptron can only create linear decision boundaries, making it unable to solve non-linearly separable problems like XOR. This limitation led to the development of multi-layer perceptrons.',
    difficulty: 'easy',
    points: 10
  },
  {
    id: 'nn-2',
    type: 'true-false',
    question: 'The universal approximation theorem states that a neural network with one hidden layer can approximate any continuous function.',
    correctAnswer: 'true',
    explanation: 'The universal approximation theorem proves that a feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of Rⁿ to arbitrary accuracy.',
    difficulty: 'medium',
    points: 15
  },
  {
    id: 'nn-3',
    type: 'fill-blank',
    question: 'Complete the perceptron learning rule equation: w(new) = w(old) + η × _____ × _____',
    blanks: ['error', 'input'],
    correctAnswer: ['error', 'input'],
    explanation: 'The perceptron learning rule updates weights based on the error (target - output) multiplied by the input value, scaled by the learning rate η.',
    difficulty: 'medium',
    points: 20
  },
  {
    id: 'nn-4',
    type: 'multiple-choice',
    question: 'Which activation function is most commonly used in hidden layers of modern neural networks?',
    options: ['Sigmoid', 'Tanh', 'ReLU', 'Linear'],
    correctAnswer: 'ReLU',
    explanation: 'ReLU (Rectified Linear Unit) is the most popular activation function for hidden layers because it helps mitigate the vanishing gradient problem and is computationally efficient.',
    difficulty: 'easy',
    points: 10
  },
  {
    id: 'nn-5',
    type: 'code-completion',
    question: 'Complete this Python function to implement the sigmoid activation function:',
    code: `def sigmoid(x):
    # Complete this function
    return ____`,
    correctAnswer: '1 / (1 + np.exp(-x))',
    explanation: 'The sigmoid function is defined as σ(x) = 1/(1 + e^(-x)). It maps any real number to a value between 0 and 1.',
    difficulty: 'medium',
    points: 25
  },
  {
    id: 'nn-6',
    type: 'multiple-choice',
    question: 'What problem does the XOR gate demonstrate about perceptrons?',
    options: [
      'Perceptrons are too slow for complex problems',
      'Single perceptrons cannot solve non-linearly separable problems',
      'Perceptrons require too much memory',
      'Perceptrons cannot handle multiple inputs'
    ],
    correctAnswer: 'Single perceptrons cannot solve non-linearly separable problems',
    explanation: 'The XOR problem cannot be solved by a single perceptron because it requires a non-linear decision boundary. This limitation led to the development of multi-layer perceptrons.',
    difficulty: 'medium',
    points: 15
  },
  {
    id: 'nn-7',
    type: 'true-false',
    question: 'Backpropagation is used to compute gradients efficiently in neural networks.',
    correctAnswer: 'true',
    explanation: 'Backpropagation uses the chain rule to efficiently compute gradients of the loss function with respect to all weights in the network, enabling gradient-based optimization.',
    difficulty: 'easy',
    points: 10
  },
  {
    id: 'nn-8',
    type: 'multiple-choice',
    question: 'Which of the following best describes the vanishing gradient problem?',
    options: [
      'Gradients become too large during training',
      'Gradients become very small in deep networks, slowing learning',
      'The network forgets previous training examples',
      'The loss function becomes non-differentiable'
    ],
    correctAnswer: 'Gradients become very small in deep networks, slowing learning',
    explanation: 'The vanishing gradient problem occurs when gradients become exponentially smaller as they propagate backward through deep networks, making it difficult to train early layers effectively.',
    difficulty: 'hard',
    points: 20
  }
]

const cnnQuiz: QuizQuestion[] = [
  {
    id: 'cnn-1',
    type: 'multiple-choice',
    question: 'What is the primary advantage of using convolution operations in CNNs?',
    options: [
      'Faster computation',
      'Parameter sharing and translation invariance',
      'Better memory usage',
      'Simpler architecture'
    ],
    correctAnswer: 'Parameter sharing and translation invariance',
    explanation: 'Convolution operations enable parameter sharing across spatial locations and provide translation invariance, meaning the network can detect features regardless of their position in the input.',
    difficulty: 'medium',
    points: 15
  },
  {
    id: 'cnn-2',
    type: 'fill-blank',
    question: 'A convolution operation with a 3×3 kernel, stride 1, and padding 1 applied to a 32×32 input produces an output of size _____×_____.',
    blanks: ['32', '32'],
    correctAnswer: ['32', '32'],
    explanation: 'With padding=1 and stride=1, the output size equals the input size. Formula: output_size = (input_size + 2×padding - kernel_size) / stride + 1 = (32 + 2×1 - 3) / 1 + 1 = 32.',
    difficulty: 'medium',
    points: 20
  },
  {
    id: 'cnn-3',
    type: 'true-false',
    question: 'Max pooling reduces the spatial dimensions of feature maps while retaining the most important information.',
    correctAnswer: 'true',
    explanation: 'Max pooling downsamples feature maps by taking the maximum value in each pooling window, reducing spatial dimensions while preserving the strongest activations.',
    difficulty: 'easy',
    points: 10
  },
  {
    id: 'cnn-4',
    type: 'multiple-choice',
    question: 'Which CNN architecture introduced residual connections?',
    options: ['AlexNet', 'VGGNet', 'ResNet', 'LeNet'],
    correctAnswer: 'ResNet',
    explanation: 'ResNet (Residual Network) introduced skip connections that allow gradients to flow directly through the network, enabling training of very deep networks.',
    difficulty: 'medium',
    points: 15
  },
  {
    id: 'cnn-5',
    type: 'code-completion',
    question: 'Complete this PyTorch code to define a basic CNN layer:',
    code: `import torch.nn as nn

class BasicCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = ____
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        return x`,
    correctAnswer: 'nn.ReLU()',
    explanation: 'nn.ReLU() creates a ReLU activation function layer. ReLU is the most commonly used activation function in CNNs.',
    difficulty: 'easy',
    points: 15
  }
]

const rnnQuiz: QuizQuestion[] = [
  {
    id: 'rnn-1',
    type: 'multiple-choice',
    question: 'What is the main advantage of RNNs over feedforward networks?',
    options: [
      'Faster training',
      'Ability to process sequential data',
      'Fewer parameters',
      'Better accuracy'
    ],
    correctAnswer: 'Ability to process sequential data',
    explanation: 'RNNs can process sequences of varying lengths and maintain memory of previous inputs through their recurrent connections, making them ideal for sequential data.',
    difficulty: 'easy',
    points: 10
  },
  {
    id: 'rnn-2',
    type: 'true-false',
    question: 'LSTM networks were designed to solve the vanishing gradient problem in traditional RNNs.',
    correctAnswer: 'true',
    explanation: 'LSTM (Long Short-Term Memory) networks use gating mechanisms to control information flow, allowing gradients to flow more effectively and enabling learning of long-term dependencies.',
    difficulty: 'medium',
    points: 15
  },
  {
    id: 'rnn-3',
    type: 'multiple-choice',
    question: 'Which gate in an LSTM cell controls what information to forget?',
    options: ['Input gate', 'Forget gate', 'Output gate', 'Cell gate'],
    correctAnswer: 'Forget gate',
    explanation: 'The forget gate determines what information should be discarded from the cell state, allowing the LSTM to forget irrelevant information.',
    difficulty: 'medium',
    points: 15
  },
  {
    id: 'rnn-4',
    type: 'fill-blank',
    question: 'In a many-to-one RNN architecture, we have _____ inputs and _____ output(s).',
    blanks: ['multiple', 'one'],
    correctAnswer: ['multiple', 'one'],
    explanation: 'Many-to-one RNNs process a sequence of inputs and produce a single output, commonly used for tasks like sentiment analysis or sequence classification.',
    difficulty: 'easy',
    points: 10
  }
]

const transformersQuiz: QuizQuestion[] = [
  {
    id: 'trans-1',
    type: 'multiple-choice',
    question: 'What is the key innovation of the Transformer architecture?',
    options: [
      'Convolutional layers',
      'Self-attention mechanism',
      'Recurrent connections',
      'Pooling operations'
    ],
    correctAnswer: 'Self-attention mechanism',
    explanation: 'The Transformer introduced self-attention, allowing the model to weigh the importance of different parts of the input sequence when processing each element.',
    difficulty: 'medium',
    points: 15
  },
  {
    id: 'trans-2',
    type: 'true-false',
    question: 'Transformers can process all positions in a sequence in parallel, unlike RNNs.',
    correctAnswer: 'true',
    explanation: 'Unlike RNNs which process sequences sequentially, Transformers can process all positions simultaneously due to their self-attention mechanism, enabling better parallelization.',
    difficulty: 'medium',
    points: 15
  },
  {
    id: 'trans-3',
    type: 'multiple-choice',
    question: 'What is the purpose of positional encoding in Transformers?',
    options: [
      'To reduce computational complexity',
      'To provide sequence order information',
      'To normalize inputs',
      'To prevent overfitting'
    ],
    correctAnswer: 'To provide sequence order information',
    explanation: 'Since Transformers process all positions in parallel, positional encodings are added to give the model information about the relative or absolute position of tokens in the sequence.',
    difficulty: 'medium',
    points: 20
  }
]

// Additional Quiz Data for All Lessons
const advancedQuiz: QuizQuestion[] = [
  {
    id: 'adv-1',
    type: 'multiple-choice',
    question: 'What is the main idea behind Generative Adversarial Networks (GANs)?',
    options: [
      'Two networks compete against each other',
      'Multiple networks work together',
      'A single network generates and discriminates',
      'Networks share parameters'
    ],
    correctAnswer: 'Two networks compete against each other',
    explanation: 'GANs consist of a generator and discriminator that compete in a minimax game, where the generator tries to fool the discriminator while the discriminator tries to distinguish real from fake data.',
    difficulty: 'medium',
    points: 15
  },
  {
    id: 'adv-2',
    type: 'true-false',
    question: 'Autoencoders are primarily used for supervised learning tasks.',
    correctAnswer: 'false',
    explanation: 'Autoencoders are unsupervised learning models that learn to compress and reconstruct data without requiring labeled examples.',
    difficulty: 'easy',
    points: 10
  },
  {
    id: 'adv-3',
    type: 'multiple-choice',
    question: 'What is the key advantage of diffusion models over GANs?',
    options: [
      'Faster training',
      'More stable training process',
      'Smaller model size',
      'Better computational efficiency'
    ],
    correctAnswer: 'More stable training process',
    explanation: 'Diffusion models offer more stable training compared to GANs, which can suffer from mode collapse and training instability.',
    difficulty: 'hard',
    points: 20
  }
]

const optimizationQuiz: QuizQuestion[] = [
  {
    id: 'opt-1',
    type: 'multiple-choice',
    question: 'Which optimizer adapts the learning rate for each parameter individually?',
    options: ['SGD', 'Adam', 'Momentum', 'RMSprop'],
    correctAnswer: 'Adam',
    explanation: 'Adam (Adaptive Moment Estimation) maintains separate learning rates for each parameter and adapts them based on first and second moment estimates.',
    difficulty: 'medium',
    points: 15
  },
  {
    id: 'opt-2',
    type: 'fill-blank',
    question: 'The learning rate schedule that reduces the learning rate by a factor when the validation loss plateaus is called _____ scheduling.',
    blanks: ['plateau'],
    correctAnswer: ['plateau'],
    explanation: 'Plateau scheduling monitors a metric (like validation loss) and reduces the learning rate when no improvement is observed for a specified number of epochs.',
    difficulty: 'medium',
    points: 15
  },
  {
    id: 'opt-3',
    type: 'true-false',
    question: 'Batch normalization helps reduce internal covariate shift.',
    correctAnswer: 'true',
    explanation: 'Batch normalization normalizes inputs to each layer, reducing internal covariate shift and allowing for higher learning rates and faster training.',
    difficulty: 'medium',
    points: 15
  }
]

const regularizationQuiz: QuizQuestion[] = [
  {
    id: 'reg-1',
    type: 'multiple-choice',
    question: 'What percentage of neurons does dropout typically disable during training?',
    options: ['10-20%', '30-50%', '60-80%', '90-95%'],
    correctAnswer: '30-50%',
    explanation: 'Dropout typically disables 30-50% of neurons during training, with 50% being a common choice for fully connected layers.',
    difficulty: 'easy',
    points: 10
  },
  {
    id: 'reg-2',
    type: 'true-false',
    question: 'L2 regularization tends to produce sparse weight matrices.',
    correctAnswer: 'false',
    explanation: 'L1 regularization produces sparse weights by driving some weights to exactly zero, while L2 regularization shrinks weights towards zero but rarely makes them exactly zero.',
    difficulty: 'medium',
    points: 15
  },
  {
    id: 'reg-3',
    type: 'fill-blank',
    question: 'Early stopping monitors the _____ loss and stops training when it starts to increase.',
    blanks: ['validation'],
    correctAnswer: ['validation'],
    explanation: 'Early stopping monitors validation loss to detect when the model starts overfitting to the training data and stops training to prevent further overfitting.',
    difficulty: 'easy',
    points: 10
  }
]

const applicationsQuiz: QuizQuestion[] = [
  {
    id: 'app-1',
    type: 'multiple-choice',
    question: 'Which deep learning application has had the most impact on autonomous vehicles?',
    options: ['Natural language processing', 'Computer vision', 'Speech recognition', 'Recommendation systems'],
    correctAnswer: 'Computer vision',
    explanation: 'Computer vision, particularly object detection and semantic segmentation, is crucial for autonomous vehicles to understand their environment.',
    difficulty: 'easy',
    points: 10
  },
  {
    id: 'app-2',
    type: 'true-false',
    question: 'Deep learning models in healthcare can achieve superhuman performance in some diagnostic tasks.',
    correctAnswer: 'true',
    explanation: 'Deep learning models have achieved superhuman performance in several medical imaging tasks, such as diabetic retinopathy detection and skin cancer classification.',
    difficulty: 'medium',
    points: 15
  },
  {
    id: 'app-3',
    type: 'multiple-choice',
    question: 'What is the main challenge in applying deep learning to drug discovery?',
    options: [
      'Lack of computational power',
      'Limited and expensive data',
      'Simple molecular structures',
      'Fast experimental validation'
    ],
    correctAnswer: 'Limited and expensive data',
    explanation: 'Drug discovery faces challenges with limited, expensive, and noisy data, making it difficult to train robust deep learning models.',
    difficulty: 'hard',
    points: 20
  }
]

// Quiz-enabled lesson components
function NeuralNetworksLessonWithQuiz({ setActiveLesson }: LessonProps) {
  const [activeTab, setActiveTab] = useState('introduction')
  const [showCode, setShowCode] = useState(false)
  const [selectedExample, setSelectedExample] = useState('perceptron')
  const [showQuiz, setShowQuiz] = useState(false)
  const [quizScore, setQuizScore] = useState<number | null>(null)

  const tabs = [
    { id: 'introduction', label: 'Introduction', icon: '📚' },
    { id: 'perceptron', label: 'Perceptron', icon: '🔵' },
    { id: 'mlp', label: 'Multi-Layer Networks', icon: '🧠' },
    { id: 'backprop', label: 'Backpropagation', icon: '🔄' },
    { id: 'activation', label: 'Activation Functions', icon: '📈' },
    { id: 'implementation', label: 'Implementation', icon: '💻' },
    { id: 'exercises', label: 'Exercises', icon: '🎯' },
    { id: 'quiz', label: 'Quiz', icon: '🧪' }
  ]

  const handleQuizComplete = (score: number, totalPoints: number) => {
    setQuizScore(Math.round((score / totalPoints) * 100))
  }

  if (showQuiz) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setShowQuiz(false)}
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-800"
          >
            <span>←</span>
            <span>Back to Lesson</span>
          </button>
          {quizScore !== null && (
            <div className="bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm">
              Last Score: {quizScore}%
            </div>
          )}
        </div>
        
        <Quiz 
          lessonId="neural-networks"
          questions={neuralNetworksQuiz}
          onComplete={handleQuizComplete}
        />
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Neural Networks Fundamentals</h2>
        <p className="text-xl text-gray-600 mb-6">
          Master the building blocks of deep learning: from single neurons to complex multi-layer networks. 
          Understand the mathematical foundations, implementation details, and practical applications.
        </p>
        <div className="flex flex-wrap gap-3">
          <span className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm">🎯 Learning Objectives</span>
          <span className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm">📊 Mathematical Foundations</span>
          <span className="bg-purple-100 text-purple-800 px-3 py-1 rounded-full text-sm">💻 Code Implementation</span>
          {quizScore !== null && (
            <span className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm">
              🧪 Quiz Score: {quizScore}%
            </span>
          )}
        </div>
      </div>

      {/* Enhanced Tab Navigation */}
      <div className="flex flex-wrap gap-2 bg-gray-100 rounded-lg p-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => {
              if (tab.id === 'quiz') {
                setShowQuiz(true)
              } else {
                setActiveTab(tab.id)
              }
            }}
            className={`flex items-center space-x-2 px-4 py-3 rounded-md text-sm font-medium transition-all duration-200 ${
              activeTab === tab.id 
                ? 'bg-white text-purple-600 shadow-sm transform scale-105' 
                : 'text-gray-600 hover:text-purple-600 hover:bg-gray-50'
            }`}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
            {tab.id === 'quiz' && quizScore !== null && (
              <span className="bg-green-500 text-white text-xs px-2 py-1 rounded-full">
                {quizScore}%
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Tab Content - Using existing content from NeuralNetworksLesson */}
      {activeTab === 'introduction' && (
        <div className="space-y-6">
          <div className="bg-blue-50 rounded-lg p-6">
            <h3 className="text-2xl font-semibold text-blue-900 mb-4">What are Neural Networks?</h3>
            <p className="text-blue-800 text-lg mb-4">
              Neural networks are computational models inspired by the human brain's structure and function. 
              They consist of interconnected nodes (neurons) that process information through weighted connections.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-blue-900 mb-3">Key Characteristics:</h4>
                <ul className="space-y-2 text-blue-800 text-sm">
                  <li>• Parallel processing</li>
                  <li>• Learning from data</li>
                  <li>• Non-linear mapping</li>
                  <li>• Fault tolerance</li>
                </ul>
              </div>
              <div className="bg-white rounded-lg p-4">
                <h4 className="font-semibold text-blue-900 mb-3">Applications:</h4>
                <ul className="space-y-2 text-blue-800 text-sm">
                  <li>• Computer vision</li>
                  <li>• Natural language processing</li>
                  <li>• Speech recognition</li>
                  <li>• Game playing</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Navigation */}
      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <div className="text-sm text-gray-500">
          Lesson 1 of 8 • Neural Networks Fundamentals
        </div>
        <div className="flex space-x-4">
          <button
            onClick={() => setShowQuiz(true)}
            className="px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-all duration-200"
          >
            Take Quiz 🧪
          </button>
          <button 
            onClick={() => setActiveLesson('cnn')}
            className="px-8 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all duration-200 transform hover:scale-105 shadow-lg"
          >
            Next: Convolutional Networks →
          </button>
        </div>
      </div>
    </div>
  )
}

function CNNLessonWithQuiz({ setActiveLesson }: LessonProps) {
  const [showQuiz, setShowQuiz] = useState(false)
  const [quizScore, setQuizScore] = useState<number | null>(null)

  const handleQuizComplete = (score: number, totalPoints: number) => {
    setQuizScore(Math.round((score / totalPoints) * 100))
  }

  if (showQuiz) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setShowQuiz(false)}
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-800"
          >
            <span>←</span>
            <span>Back to Lesson</span>
          </button>
          {quizScore !== null && (
            <div className="bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm">
              Last Score: {quizScore}%
            </div>
          )}
        </div>
        
        <Quiz 
          lessonId="cnn"
          questions={cnnQuiz}
          onComplete={handleQuizComplete}
        />
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-blue-50 to-green-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Convolutional Neural Networks</h2>
        <p className="text-xl text-gray-600 mb-6">
          Master CNNs for computer vision tasks. Learn convolution operations, pooling, and modern architectures 
          that revolutionized image processing and pattern recognition.
        </p>
        {quizScore !== null && (
          <div className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm inline-block">
            🧪 Quiz Score: {quizScore}%
          </div>
        )}
      </div>

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('neural-networks')}
          className="px-6 py-2 text-gray-600 hover:text-blue-600 transition-colors"
        >
          ← Previous: Neural Networks
        </button>
        <div className="text-sm text-gray-500">Lesson 2 of 8 • Convolutional Networks</div>
        <div className="flex space-x-4">
          <button
            onClick={() => setShowQuiz(true)}
            className="px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-all duration-200"
          >
            Take Quiz 🧪
          </button>
          <button 
            onClick={() => setActiveLesson('rnn')}
            className="px-8 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all duration-200"
          >
            Next: Recurrent Networks →
          </button>
        </div>
      </div>
    </div>
  )
}

function RNNLessonWithQuiz({ setActiveLesson }: LessonProps) {
  const [showQuiz, setShowQuiz] = useState(false)
  const [quizScore, setQuizScore] = useState<number | null>(null)

  const handleQuizComplete = (score: number, totalPoints: number) => {
    setQuizScore(Math.round((score / totalPoints) * 100))
  }

  if (showQuiz) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setShowQuiz(false)}
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-800"
          >
            <span>←</span>
            <span>Back to Lesson</span>
          </button>
          {quizScore !== null && (
            <div className="bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm">
              Last Score: {quizScore}%
            </div>
          )}
        </div>
        
        <Quiz 
          lessonId="rnn"
          questions={rnnQuiz}
          onComplete={handleQuizComplete}
        />
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-green-50 to-purple-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Recurrent Neural Networks</h2>
        <p className="text-xl text-gray-600 mb-6">
          Learn RNNs for sequential data processing. Understand LSTM, GRU, and how to handle 
          time series, natural language, and other sequential patterns.
        </p>
        {quizScore !== null && (
          <div className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm inline-block">
            🧪 Quiz Score: {quizScore}%
          </div>
        )}
      </div>

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('cnn')}
          className="px-6 py-2 text-gray-600 hover:text-green-600 transition-colors"
        >
          ← Previous: CNNs
        </button>
        <div className="text-sm text-gray-500">Lesson 3 of 8 • Recurrent Networks</div>
        <div className="flex space-x-4">
          <button
            onClick={() => setShowQuiz(true)}
            className="px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-all duration-200"
          >
            Take Quiz 🧪
          </button>
          <button 
            onClick={() => setActiveLesson('transformers')}
            className="px-8 py-3 bg-gradient-to-r from-green-600 to-purple-600 text-white rounded-lg hover:from-green-700 hover:to-purple-700 transition-all duration-200"
          >
            Next: Transformers →
          </button>
        </div>
      </div>
    </div>
  )
}

function TransformersLessonWithQuiz({ setActiveLesson }: LessonProps) {
  const [showQuiz, setShowQuiz] = useState(false)
  const [quizScore, setQuizScore] = useState<number | null>(null)

  const handleQuizComplete = (score: number, totalPoints: number) => {
    setQuizScore(Math.round((score / totalPoints) * 100))
  }

  if (showQuiz) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setShowQuiz(false)}
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-800"
          >
            <span>←</span>
            <span>Back to Lesson</span>
          </button>
          {quizScore !== null && (
            <div className="bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm">
              Last Score: {quizScore}%
            </div>
          )}
        </div>
        
        <Quiz 
          lessonId="transformers"
          questions={transformersQuiz}
          onComplete={handleQuizComplete}
        />
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Transformers & Attention</h2>
        <p className="text-xl text-gray-600 mb-6">
          Explore the revolutionary Transformer architecture that powers modern NLP. Learn attention 
          mechanisms, self-attention, and how transformers changed AI forever.
        </p>
        {quizScore !== null && (
          <div className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm inline-block">
            🧪 Quiz Score: {quizScore}%
          </div>
        )}
      </div>

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('rnn')}
          className="px-6 py-2 text-gray-600 hover:text-purple-600 transition-colors"
        >
          ← Previous: RNNs
        </button>
        <div className="text-sm text-gray-500">Lesson 4 of 8 • Transformers & Attention</div>
        <div className="flex space-x-4">
          <button
            onClick={() => setShowQuiz(true)}
            className="px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-all duration-200"
          >
            Take Quiz 🧪
          </button>
          <button 
            onClick={() => setActiveLesson('advanced')}
            className="px-8 py-3 bg-gradient-to-r from-purple-600 to-red-600 text-white rounded-lg hover:from-purple-700 hover:to-red-700 transition-all duration-200"
          >
            Next: Advanced Architectures →
          </button>
        </div>
      </div>
    </div>
  )
}

function AdvancedLessonWithQuiz({ setActiveLesson }: LessonProps) {
  const [showQuiz, setShowQuiz] = useState(false)
  const [quizScore, setQuizScore] = useState<number | null>(null)

  const handleQuizComplete = (score: number, totalPoints: number) => {
    setQuizScore(Math.round((score / totalPoints) * 100))
  }

  if (showQuiz) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setShowQuiz(false)}
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-800"
          >
            <span>←</span>
            <span>Back to Lesson</span>
          </button>
          {quizScore !== null && (
            <div className="bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm">
              Last Score: {quizScore}%
            </div>
          )}
        </div>
        
        <Quiz 
          lessonId="advanced"
          questions={advancedQuiz}
          onComplete={handleQuizComplete}
        />
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-red-50 to-orange-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Advanced Architectures</h2>
        <p className="text-xl text-gray-600 mb-6">
          Explore cutting-edge neural network architectures including GANs, autoencoders, 
          diffusion models, and neural fields that push the boundaries of AI.
        </p>
        {quizScore !== null && (
          <div className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm inline-block">
            🧪 Quiz Score: {quizScore}%
          </div>
        )}
      </div>

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('transformers')}
          className="px-6 py-2 text-gray-600 hover:text-red-600 transition-colors"
        >
          ← Previous: Transformers
        </button>
        <div className="text-sm text-gray-500">Lesson 5 of 8 • Advanced Architectures</div>
        <div className="flex space-x-4">
          <button
            onClick={() => setShowQuiz(true)}
            className="px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-all duration-200"
          >
            Take Quiz 🧪
          </button>
          <button 
            onClick={() => setActiveLesson('optimization')}
            className="px-8 py-3 bg-gradient-to-r from-red-600 to-yellow-600 text-white rounded-lg hover:from-red-700 hover:to-yellow-700 transition-all duration-200"
          >
            Next: Training & Optimization →
          </button>
        </div>
      </div>
    </div>
  )
}

function OptimizationLessonWithQuiz({ setActiveLesson }: LessonProps) {
  const [showQuiz, setShowQuiz] = useState(false)
  const [quizScore, setQuizScore] = useState<number | null>(null)

  const handleQuizComplete = (score: number, totalPoints: number) => {
    setQuizScore(Math.round((score / totalPoints) * 100))
  }

  if (showQuiz) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setShowQuiz(false)}
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-800"
          >
            <span>←</span>
            <span>Back to Lesson</span>
          </button>
          {quizScore !== null && (
            <div className="bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm">
              Last Score: {quizScore}%
            </div>
          )}
        </div>
        
        <Quiz 
          lessonId="optimization"
          questions={optimizationQuiz}
          onComplete={handleQuizComplete}
        />
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-yellow-50 to-green-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Training & Optimization</h2>
        <p className="text-xl text-gray-600 mb-6">
          Master the art of training neural networks efficiently. Learn about optimizers, 
          learning rate schedules, normalization techniques, and advanced training strategies.
        </p>
        {quizScore !== null && (
          <div className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm inline-block">
            🧪 Quiz Score: {quizScore}%
          </div>
        )}
      </div>

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('advanced')}
          className="px-6 py-2 text-gray-600 hover:text-yellow-600 transition-colors"
        >
          ← Previous: Advanced Architectures
        </button>
        <div className="text-sm text-gray-500">Lesson 6 of 8 • Training & Optimization</div>
        <div className="flex space-x-4">
          <button
            onClick={() => setShowQuiz(true)}
            className="px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-all duration-200"
          >
            Take Quiz 🧪
          </button>
          <button 
            onClick={() => setActiveLesson('regularization')}
            className="px-8 py-3 bg-gradient-to-r from-yellow-600 to-blue-600 text-white rounded-lg hover:from-yellow-700 hover:to-blue-700 transition-all duration-200"
          >
            Next: Regularization →
          </button>
        </div>
      </div>
    </div>
  )
}

function RegularizationLessonWithQuiz({ setActiveLesson }: LessonProps) {
  const [showQuiz, setShowQuiz] = useState(false)
  const [quizScore, setQuizScore] = useState<number | null>(null)

  const handleQuizComplete = (score: number, totalPoints: number) => {
    setQuizScore(Math.round((score / totalPoints) * 100))
  }

  if (showQuiz) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setShowQuiz(false)}
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-800"
          >
            <span>←</span>
            <span>Back to Lesson</span>
          </button>
          {quizScore !== null && (
            <div className="bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm">
              Last Score: {quizScore}%
            </div>
          )}
        </div>
        
        <Quiz 
          lessonId="regularization"
          questions={regularizationQuiz}
          onComplete={handleQuizComplete}
        />
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Regularization Techniques</h2>
        <p className="text-xl text-gray-600 mb-6">
          Learn how to prevent overfitting and improve generalization. Master dropout, weight decay, 
          data augmentation, and other techniques that make models robust.
        </p>
        {quizScore !== null && (
          <div className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm inline-block">
            🧪 Quiz Score: {quizScore}%
          </div>
        )}
      </div>

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('optimization')}
          className="px-6 py-2 text-gray-600 hover:text-blue-600 transition-colors"
        >
          ← Previous: Optimization
        </button>
        <div className="text-sm text-gray-500">Lesson 7 of 8 • Regularization Techniques</div>
        <div className="flex space-x-4">
          <button
            onClick={() => setShowQuiz(true)}
            className="px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-all duration-200"
          >
            Take Quiz 🧪
          </button>
          <button 
            onClick={() => setActiveLesson('applications')}
            className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200"
          >
            Next: Applications →
          </button>
        </div>
      </div>
    </div>
  )
}

function ApplicationsLessonWithQuiz({ setActiveLesson }: LessonProps) {
  const [showQuiz, setShowQuiz] = useState(false)
  const [quizScore, setQuizScore] = useState<number | null>(null)

  const handleQuizComplete = (score: number, totalPoints: number) => {
    setQuizScore(Math.round((score / totalPoints) * 100))
  }

  if (showQuiz) {
    return (
      <div className="space-y-8">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setShowQuiz(false)}
            className="flex items-center space-x-2 text-gray-600 hover:text-gray-800"
          >
            <span>←</span>
            <span>Back to Lesson</span>
          </button>
          {quizScore !== null && (
            <div className="bg-green-100 text-green-800 px-4 py-2 rounded-full text-sm">
              Last Score: {quizScore}%
            </div>
          )}
        </div>
        
        <Quiz 
          lessonId="applications"
          questions={applicationsQuiz}
          onComplete={handleQuizComplete}
        />
      </div>
    )
  }

  return (
    <div className="space-y-8">
      <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg p-6">
        <h2 className="text-4xl font-bold text-gray-900 mb-4">Real-World Applications</h2>
        <p className="text-xl text-gray-600 mb-6">
          Explore how deep learning transforms industries and solves real-world problems. 
          From healthcare to autonomous systems, see AI in action across domains.
        </p>
        {quizScore !== null && (
          <div className="bg-yellow-100 text-yellow-800 px-3 py-1 rounded-full text-sm inline-block">
            🧪 Quiz Score: {quizScore}%
          </div>
        )}
      </div>

      <div className="flex justify-between items-center pt-8 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('regularization')}
          className="px-6 py-2 text-gray-600 hover:text-purple-600 transition-colors"
        >
          ← Previous: Regularization
        </button>
        <div className="text-sm text-gray-500">Lesson 8 of 8 • Real-World Applications</div>
        <div className="flex space-x-4">
          <button
            onClick={() => setShowQuiz(true)}
            className="px-6 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-all duration-200"
          >
            Take Quiz 🧪
          </button>
          <div className="px-8 py-3 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg">
            🎉 Course Complete!
          </div>
        </div>
      </div>
    </div>
  )
}