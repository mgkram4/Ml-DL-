'use client'

import { useState } from 'react'

interface LessonProps {
  setActiveLesson: (lesson: string) => void
}

export default function MachineLearningContent() {
  const [activeLesson, setActiveLesson] = useState('introduction')

  const lessons = [
    { id: 'introduction', title: 'Introduction to ML', icon: '🎯', duration: '30 min' },
    { id: 'supervised', title: 'Supervised Learning', icon: '📊', duration: '45 min' },
    { id: 'unsupervised', title: 'Unsupervised Learning', icon: '🔍', duration: '40 min' },
    { id: 'evaluation', title: 'Model Evaluation', icon: '📈', duration: '35 min' },
    { id: 'algorithms', title: 'Key Algorithms', icon: '⚙️', duration: '60 min' },
    { id: 'projects', title: 'Hands-on Projects', icon: '🛠️', duration: '90 min' }
  ]

  return (
    <section className="pt-20 pb-16 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            <span className="gradient-text">Machine Learning</span> Mastery
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Master the fundamentals of machine learning through comprehensive lessons, 
            interactive examples, and real-world projects.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Lesson Navigation */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg p-6 sticky top-24">
              <h3 className="font-semibold text-gray-900 mb-4">Course Outline</h3>
              <nav className="space-y-2">
                {lessons.map((lesson) => (
                  <button
                    key={lesson.id}
                    onClick={() => setActiveLesson(lesson.id)}
                    className={`w-full text-left px-4 py-3 rounded-lg transition-all duration-200 flex items-center space-x-3 ${
                      activeLesson === lesson.id
                        ? 'bg-blue-100 text-blue-700 border-l-4 border-blue-500'
                        : 'text-gray-600 hover:bg-gray-50 hover:text-blue-600'
                    }`}
                  >
                    <span className="text-lg">{lesson.icon}</span>
                    <div className="flex-1">
                      <div className="font-medium">{lesson.title}</div>
                      <div className="text-xs text-gray-500">{lesson.duration}</div>
                    </div>
                  </button>
                ))}
              </nav>
              
              {/* Progress */}
              <div className="mt-6 pt-6 border-t border-gray-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">Progress</span>
                  <span className="text-sm text-gray-500">2/6 Complete</span>
                </div>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: '33%' }}></div>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-xl shadow-lg p-8">
              {activeLesson === 'introduction' && <IntroductionLesson setActiveLesson={setActiveLesson} />}
              {activeLesson === 'supervised' && <SupervisedLearningLesson setActiveLesson={setActiveLesson} />}
              {activeLesson === 'unsupervised' && <UnsupervisedLearningLesson setActiveLesson={setActiveLesson} />}
              {activeLesson === 'evaluation' && <ModelEvaluationLesson setActiveLesson={setActiveLesson} />}
              {activeLesson === 'algorithms' && <AlgorithmsLesson setActiveLesson={setActiveLesson} />}
              {activeLesson === 'projects' && <ProjectsLesson setActiveLesson={setActiveLesson} />}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function IntroductionLesson({ setActiveLesson }: LessonProps) {
  const [activeExample, setActiveExample] = useState('traditional-vs-ml')
  const [selectedApplication, setSelectedApplication] = useState('email')

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Introduction to Machine Learning</h2>
        <p className="text-lg text-gray-600 mb-6">
          Welcome to the fascinating world of machine learning! This comprehensive introduction will take you through 
          the fundamental concepts, real-world applications, and core principles that power modern AI systems.
        </p>
      </div>

      {/* What is Machine Learning - Expanded */}
      <div className="bg-blue-50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-blue-900 mb-3">What is Machine Learning?</h3>
        <p className="text-blue-800 mb-4">
          Machine Learning is a subset of artificial intelligence that enables computers to learn and improve 
          from experience without being explicitly programmed. Instead of following pre-programmed instructions, 
          ML systems identify patterns in data and make predictions or decisions based on those patterns.
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div className="bg-white rounded p-4">
            <h4 className="font-semibold text-blue-900 mb-2">Traditional Programming</h4>
            <div className="font-mono text-sm mb-3">
              Input Data + Algorithm → Output
            </div>
            <p className="text-blue-800 text-sm">
              Programmers write explicit rules and logic. The computer follows these exact instructions 
              to process data and produce results.
            </p>
          </div>
          <div className="bg-white rounded p-4">
            <h4 className="font-semibold text-blue-900 mb-2">Machine Learning</h4>
            <div className="font-mono text-sm mb-3">
              Input Data + Desired Output → Algorithm
            </div>
            <p className="text-blue-800 text-sm">
              The system learns patterns from examples and creates its own algorithm to make 
              predictions on new, unseen data.
            </p>
          </div>
        </div>

        <div className="bg-white rounded p-4">
          <h4 className="font-semibold text-blue-900 mb-3">Key Characteristics of Machine Learning:</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <ul className="text-blue-800 text-sm space-y-2">
              <li>• <strong>Data-Driven:</strong> Learns from examples rather than explicit programming</li>
              <li>• <strong>Pattern Recognition:</strong> Identifies complex relationships in data</li>
              <li>• <strong>Generalization:</strong> Makes predictions on new, unseen data</li>
              <li>• <strong>Iterative Improvement:</strong> Performance improves with more data</li>
            </ul>
            <ul className="text-blue-800 text-sm space-y-2">
              <li>• <strong>Automation:</strong> Reduces need for manual rule creation</li>
              <li>• <strong>Scalability:</strong> Handles large volumes of data efficiently</li>
              <li>• <strong>Adaptability:</strong> Adjusts to new patterns and changes</li>
              <li>• <strong>Probabilistic:</strong> Provides confidence levels with predictions</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Types of Machine Learning - Greatly Expanded */}
      <div className="space-y-6">
        <h3 className="text-2xl font-bold text-gray-900">Types of Machine Learning</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
              <span className="text-2xl">🎯</span>
            </div>
            <h4 className="font-semibold text-gray-900 mb-2">Supervised Learning</h4>
            <p className="text-gray-600 text-sm mb-3">Learning with labeled examples to predict outcomes</p>
            <div className="text-xs text-gray-500 space-y-1">
              <div><strong>Examples:</strong> Email spam detection, house price prediction</div>
              <div><strong>Data:</strong> Input-output pairs (features and labels)</div>
              <div><strong>Goal:</strong> Learn mapping from inputs to outputs</div>
            </div>
          </div>
          
          <div className="border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
              <span className="text-2xl">🔍</span>
            </div>
            <h4 className="font-semibold text-gray-900 mb-2">Unsupervised Learning</h4>
            <p className="text-gray-600 text-sm mb-3">Finding hidden patterns in data without labels</p>
            <div className="text-xs text-gray-500 space-y-1">
              <div><strong>Examples:</strong> Customer segmentation, anomaly detection</div>
              <div><strong>Data:</strong> Input data only (no target labels)</div>
              <div><strong>Goal:</strong> Discover hidden structure in data</div>
            </div>
          </div>
          
          <div className="border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
            <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center mb-4">
              <span className="text-2xl">🎮</span>
            </div>
            <h4 className="font-semibold text-gray-900 mb-2">Reinforcement Learning</h4>
            <p className="text-gray-600 text-sm mb-3">Learning through trial and error with rewards</p>
            <div className="text-xs text-gray-500 space-y-1">
              <div><strong>Examples:</strong> Game playing, robotics, autonomous driving</div>
              <div><strong>Data:</strong> Actions, states, and rewards</div>
              <div><strong>Goal:</strong> Learn optimal actions to maximize rewards</div>
            </div>
          </div>
        </div>

        {/* Detailed breakdown of each type */}
        <div className="bg-gray-50 rounded-lg p-6">
          <h4 className="font-semibold text-gray-900 mb-4">Detailed Breakdown:</h4>
          
          <div className="space-y-4">
            <div className="bg-white rounded p-4">
              <h5 className="font-medium text-green-900 mb-2">Supervised Learning Subtypes:</h5>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <strong className="text-green-800">Classification:</strong>
                  <ul className="text-gray-600 mt-1 space-y-1">
                    <li>• Predicts discrete categories/classes</li>
                    <li>• Binary: Spam/Not Spam, Fraud/Legitimate</li>
                    <li>• Multi-class: Animal species, Product categories</li>
                    <li>• Multi-label: Movie genres, Medical conditions</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-green-800">Regression:</strong>
                  <ul className="text-gray-600 mt-1 space-y-1">
                    <li>• Predicts continuous numerical values</li>
                    <li>• Linear: House prices, Stock prices</li>
                    <li>• Non-linear: Complex relationships</li>
                    <li>• Time series: Weather forecasting</li>
                  </ul>
                </div>
              </div>
            </div>

            <div className="bg-white rounded p-4">
              <h5 className="font-medium text-purple-900 mb-2">Unsupervised Learning Subtypes:</h5>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <strong className="text-purple-800">Clustering:</strong>
                  <ul className="text-gray-600 mt-1 space-y-1">
                    <li>• Groups similar data points</li>
                    <li>• Customer segmentation</li>
                    <li>• Gene sequencing</li>
                    <li>• Market research</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-purple-800">Association:</strong>
                  <ul className="text-gray-600 mt-1 space-y-1">
                    <li>• Finds relationships between items</li>
                    <li>• Market basket analysis</li>
                    <li>• Recommendation systems</li>
                    <li>• Web usage patterns</li>
                  </ul>
                </div>
                <div>
                  <strong className="text-purple-800">Dimensionality Reduction:</strong>
                  <ul className="text-gray-600 mt-1 space-y-1">
                    <li>• Reduces feature complexity</li>
                    <li>• Data visualization</li>
                    <li>• Noise reduction</li>
                    <li>• Feature extraction</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Machine Learning Workflow */}
      <div className="bg-indigo-50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-indigo-900 mb-4">The Machine Learning Workflow</h3>
        <p className="text-indigo-800 mb-6">
          Understanding the complete ML workflow is crucial for successful projects. Here's the step-by-step process:
        </p>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white rounded p-4 text-center">
            <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-indigo-600 font-bold">1</span>
            </div>
            <h4 className="font-medium text-indigo-900 mb-2">Problem Definition</h4>
            <p className="text-indigo-700 text-sm">Define objectives, success metrics, and constraints</p>
          </div>
          
          <div className="bg-white rounded p-4 text-center">
            <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-indigo-600 font-bold">2</span>
            </div>
            <h4 className="font-medium text-indigo-900 mb-2">Data Collection</h4>
            <p className="text-indigo-700 text-sm">Gather relevant, quality data from various sources</p>
          </div>
          
          <div className="bg-white rounded p-4 text-center">
            <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-indigo-600 font-bold">3</span>
            </div>
            <h4 className="font-medium text-indigo-900 mb-2">Data Preparation</h4>
            <p className="text-indigo-700 text-sm">Clean, transform, and engineer features</p>
          </div>
          
          <div className="bg-white rounded p-4 text-center">
            <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-indigo-600 font-bold">4</span>
            </div>
            <h4 className="font-medium text-indigo-900 mb-2">Model Selection</h4>
            <p className="text-indigo-700 text-sm">Choose appropriate algorithms and architectures</p>
          </div>
          
          <div className="bg-white rounded p-4 text-center">
            <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-indigo-600 font-bold">5</span>
            </div>
            <h4 className="font-medium text-indigo-900 mb-2">Training</h4>
            <p className="text-indigo-700 text-sm">Train models on prepared data with optimization</p>
          </div>
          
          <div className="bg-white rounded p-4 text-center">
            <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-indigo-600 font-bold">6</span>
            </div>
            <h4 className="font-medium text-indigo-900 mb-2">Evaluation</h4>
            <p className="text-indigo-700 text-sm">Assess performance using appropriate metrics</p>
          </div>
          
          <div className="bg-white rounded p-4 text-center">
            <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-indigo-600 font-bold">7</span>
            </div>
            <h4 className="font-medium text-indigo-900 mb-2">Deployment</h4>
            <p className="text-indigo-700 text-sm">Deploy model to production environment</p>
          </div>
          
          <div className="bg-white rounded p-4 text-center">
            <div className="w-12 h-12 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-indigo-600 font-bold">8</span>
            </div>
            <h4 className="font-medium text-indigo-900 mb-2">Monitoring</h4>
            <p className="text-indigo-700 text-sm">Monitor performance and retrain as needed</p>
          </div>
        </div>
      </div>

      {/* Real-World Applications - Greatly Expanded */}
      <div className="bg-gray-50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Real-World Applications</h3>
        <p className="text-gray-700 mb-6">
          Machine learning is transforming industries and powering innovations across every sector. 
          Explore these comprehensive examples to understand ML's impact:
        </p>
        
        {/* Application Selector */}
        <div className="flex flex-wrap gap-2 mb-6">
          {[
            { id: 'email', label: 'Email & Communication', icon: '📧' },
            { id: 'ecommerce', label: 'E-commerce', icon: '🛒' },
            { id: 'transportation', label: 'Transportation', icon: '🚗' },
            { id: 'healthcare', label: 'Healthcare', icon: '🏥' },
            { id: 'finance', label: 'Finance', icon: '💰' },
            { id: 'entertainment', label: 'Entertainment', icon: '🎬' }
          ].map((app) => (
            <button
              key={app.id}
              onClick={() => setSelectedApplication(app.id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                selectedApplication === app.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-blue-50'
              }`}
            >
              {app.icon} {app.label}
            </button>
          ))}
        </div>

        {/* Application Details */}
        <div className="bg-white rounded-lg p-6">
          {selectedApplication === 'email' && (
            <div>
              <h4 className="font-semibold text-gray-900 mb-3">📧 Email & Communication</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h5 className="font-medium text-gray-800 mb-2">Spam Detection</h5>
                  <ul className="text-gray-600 text-sm space-y-1 mb-4">
                    <li>• Analyzes email content, sender reputation, and metadata</li>
                    <li>• Uses NLP to identify suspicious patterns and keywords</li>
                    <li>• Continuously learns from user feedback</li>
                    <li>• Achieves 99%+ accuracy in modern systems</li>
                  </ul>
                  
                  <h5 className="font-medium text-gray-800 mb-2">Smart Compose & Reply</h5>
                  <ul className="text-gray-600 text-sm space-y-1">
                    <li>• Predicts and suggests email completions</li>
                    <li>• Generates contextually appropriate responses</li>
                    <li>• Learns from individual writing patterns</li>
                    <li>• Saves millions of hours of typing time</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium text-gray-800 mb-2">Language Translation</h5>
                  <ul className="text-gray-600 text-sm space-y-1 mb-4">
                    <li>• Real-time translation of emails and messages</li>
                    <li>• Supports 100+ languages with high accuracy</li>
                    <li>• Preserves context and tone</li>
                    <li>• Enables global communication</li>
                  </ul>
                  
                  <h5 className="font-medium text-gray-800 mb-2">Priority Inbox</h5>
                  <ul className="text-gray-600 text-sm space-y-1">
                    <li>• Automatically categorizes email importance</li>
                    <li>• Learns from user behavior and preferences</li>
                    <li>• Reduces information overload</li>
                    <li>• Improves productivity and focus</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {selectedApplication === 'ecommerce' && (
            <div>
              <h4 className="font-semibold text-gray-900 mb-3">🛒 E-commerce & Retail</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h5 className="font-medium text-gray-800 mb-2">Recommendation Systems</h5>
                  <ul className="text-gray-600 text-sm space-y-1 mb-4">
                    <li>• Collaborative filtering based on user behavior</li>
                    <li>• Content-based recommendations using item features</li>
                    <li>• Hybrid approaches combining multiple methods</li>
                    <li>• Drives 35% of Amazon's revenue</li>
                  </ul>
                  
                  <h5 className="font-medium text-gray-800 mb-2">Dynamic Pricing</h5>
                  <ul className="text-gray-600 text-sm space-y-1">
                    <li>• Real-time price optimization based on demand</li>
                    <li>• Competitor price monitoring and adjustment</li>
                    <li>• Inventory level considerations</li>
                    <li>• Maximizes revenue and profit margins</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium text-gray-800 mb-2">Fraud Detection</h5>
                  <ul className="text-gray-600 text-sm space-y-1 mb-4">
                    <li>• Analyzes transaction patterns and anomalies</li>
                    <li>• Real-time risk scoring for payments</li>
                    <li>• Reduces false positives while catching fraud</li>
                    <li>• Saves billions in fraudulent transactions</li>
                  </ul>
                  
                  <h5 className="font-medium text-gray-800 mb-2">Inventory Management</h5>
                  <ul className="text-gray-600 text-sm space-y-1">
                    <li>• Demand forecasting and stock optimization</li>
                    <li>• Automated reordering and supply chain management</li>
                    <li>• Seasonal trend analysis and planning</li>
                    <li>• Reduces waste and stockouts</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {selectedApplication === 'healthcare' && (
            <div>
              <h4 className="font-semibold text-gray-900 mb-3">🏥 Healthcare & Medicine</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h5 className="font-medium text-gray-800 mb-2">Medical Imaging</h5>
                  <ul className="text-gray-600 text-sm space-y-1 mb-4">
                    <li>• Cancer detection in X-rays, MRIs, and CT scans</li>
                    <li>• Diabetic retinopathy screening from eye images</li>
                    <li>• Skin cancer identification from photographs</li>
                    <li>• Often exceeds human radiologist accuracy</li>
                  </ul>
                  
                  <h5 className="font-medium text-gray-800 mb-2">Drug Discovery</h5>
                  <ul className="text-gray-600 text-sm space-y-1">
                    <li>• Molecular property prediction and optimization</li>
                    <li>• Target identification and validation</li>
                    <li>• Clinical trial optimization and patient matching</li>
                    <li>• Reduces development time from 15 to 5 years</li>
                  </ul>
                </div>
                <div>
                  <h5 className="font-medium text-gray-800 mb-2">Personalized Treatment</h5>
                  <ul className="text-gray-600 text-sm space-y-1 mb-4">
                    <li>• Genomic analysis for targeted therapies</li>
                    <li>• Treatment response prediction</li>
                    <li>• Dosage optimization based on patient factors</li>
                    <li>• Improves outcomes while reducing side effects</li>
                  </ul>
                  
                  <h5 className="font-medium text-gray-800 mb-2">Epidemic Monitoring</h5>
                  <ul className="text-gray-600 text-sm space-y-1">
                    <li>• Disease outbreak prediction and tracking</li>
                    <li>• Contact tracing and transmission modeling</li>
                    <li>• Resource allocation and capacity planning</li>
                    <li>• Early warning systems for public health</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* Add similar detailed sections for other applications */}
        </div>
      </div>

      {/* Key Concepts and Terminology */}
      <div className="bg-yellow-50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-yellow-900 mb-4">Essential ML Terminology</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-4">
            <div className="bg-white rounded p-4">
              <h4 className="font-medium text-yellow-900 mb-2">Data Terms</h4>
              <div className="space-y-2 text-sm">
                <div><strong>Dataset:</strong> Collection of data used for training</div>
                <div><strong>Features:</strong> Input variables or attributes</div>
                <div><strong>Labels:</strong> Target outputs or ground truth</div>
                <div><strong>Training Set:</strong> Data used to train the model</div>
                <div><strong>Test Set:</strong> Data used to evaluate final performance</div>
                <div><strong>Validation Set:</strong> Data used for model selection</div>
              </div>
            </div>
            
            <div className="bg-white rounded p-4">
              <h4 className="font-medium text-yellow-900 mb-2">Model Terms</h4>
              <div className="space-y-2 text-sm">
                <div><strong>Algorithm:</strong> The method used to learn patterns</div>
                <div><strong>Model:</strong> The trained algorithm ready for predictions</div>
                <div><strong>Parameters:</strong> Internal variables learned during training</div>
                <div><strong>Hyperparameters:</strong> Settings that control learning</div>
                <div><strong>Weights:</strong> Numerical values that determine importance</div>
                <div><strong>Bias:</strong> Constant term added to predictions</div>
              </div>
            </div>
          </div>
          
          <div className="space-y-4">
            <div className="bg-white rounded p-4">
              <h4 className="font-medium text-yellow-900 mb-2">Performance Terms</h4>
              <div className="space-y-2 text-sm">
                <div><strong>Accuracy:</strong> Percentage of correct predictions</div>
                <div><strong>Precision:</strong> True positives / (True + False positives)</div>
                <div><strong>Recall:</strong> True positives / (True positives + False negatives)</div>
                <div><strong>F1-Score:</strong> Harmonic mean of precision and recall</div>
                <div><strong>Loss Function:</strong> Measures prediction errors</div>
                <div><strong>Cross-Validation:</strong> Technique to assess generalization</div>
              </div>
            </div>
            
            <div className="bg-white rounded p-4">
              <h4 className="font-medium text-yellow-900 mb-2">Problem Terms</h4>
              <div className="space-y-2 text-sm">
                <div><strong>Overfitting:</strong> Model memorizes training data</div>
                <div><strong>Underfitting:</strong> Model is too simple to capture patterns</div>
                <div><strong>Generalization:</strong> Performance on new, unseen data</div>
                <div><strong>Feature Engineering:</strong> Creating useful input variables</div>
                <div><strong>Regularization:</strong> Techniques to prevent overfitting</div>
                <div><strong>Ensemble:</strong> Combining multiple models</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Prerequisites and Next Steps */}
      <div className="bg-green-50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-green-900 mb-4">Prerequisites & Learning Path</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-green-900 mb-3">Mathematical Prerequisites</h4>
            <div className="space-y-3">
              <div className="bg-white rounded p-3">
                <h5 className="font-medium text-green-800 mb-1">Statistics & Probability</h5>
                <p className="text-green-700 text-sm">Mean, variance, distributions, hypothesis testing</p>
              </div>
              <div className="bg-white rounded p-3">
                <h5 className="font-medium text-green-800 mb-1">Linear Algebra</h5>
                <p className="text-green-700 text-sm">Vectors, matrices, eigenvalues, matrix operations</p>
              </div>
              <div className="bg-white rounded p-3">
                <h5 className="font-medium text-green-800 mb-1">Calculus</h5>
                <p className="text-green-700 text-sm">Derivatives, gradients, optimization basics</p>
              </div>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-green-900 mb-3">Programming Skills</h4>
            <div className="space-y-3">
              <div className="bg-white rounded p-3">
                <h5 className="font-medium text-green-800 mb-1">Python Programming</h5>
                <p className="text-green-700 text-sm">Data structures, functions, object-oriented programming</p>
              </div>
              <div className="bg-white rounded p-3">
                <h5 className="font-medium text-green-800 mb-1">Data Manipulation</h5>
                <p className="text-green-700 text-sm">Pandas, NumPy for data processing and analysis</p>
              </div>
              <div className="bg-white rounded p-3">
                <h5 className="font-medium text-green-800 mb-1">Visualization</h5>
                <p className="text-green-700 text-sm">Matplotlib, Seaborn for data exploration</p>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-6 p-4 bg-white rounded">
          <h4 className="font-medium text-green-900 mb-2">💡 Don't worry if you're missing some prerequisites!</h4>
          <p className="text-green-700 text-sm">
            This course is designed to be accessible to beginners. We'll introduce mathematical concepts 
            as needed and provide practical examples that build intuition. You can always dive deeper 
            into the math later as your interest and needs develop.
          </p>
        </div>
      </div>

      <div className="flex justify-between items-center pt-6 border-t border-gray-200">
        <div className="text-sm text-gray-500">Lesson 1 of 6 • Estimated time: 45 minutes</div>
        <button 
          onClick={() => setActiveLesson('supervised')}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Next: Supervised Learning →
        </button>
      </div>
    </div>
  )
}

function SupervisedLearningLesson({ setActiveLesson }: LessonProps) {
  const [activeTab, setActiveTab] = useState('regression')
  const [houseSize, setHouseSize] = useState(2000)
  const [bedrooms, setBedrooms] = useState(3)
  const [houseAge, setHouseAge] = useState(10)
  const [emailContent, setEmailContent] = useState("Congratulations! You've won $1,000,000! Click here to claim your prize now!")
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('linear')

  // Calculate house price prediction
  const predictedPrice = 150 * houseSize + 25000 * bedrooms - 2000 * houseAge + 50000

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Supervised Learning</h2>
        <p className="text-lg text-gray-600 mb-6">
          Supervised learning is the most common type of machine learning, where we learn from labeled examples 
          to make predictions on new data. Think of it as learning with a teacher who provides the correct answers 
          during training, so the model can learn to make accurate predictions on its own.
        </p>
      </div>

      {/* Core Concepts */}
      <div className="bg-blue-50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-blue-900 mb-4">Core Concepts of Supervised Learning</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white rounded p-4">
            <h4 className="font-semibold text-blue-900 mb-3">Training Process</h4>
            <div className="space-y-2 text-sm text-blue-800">
              <div className="flex items-start space-x-2">
                <span className="font-bold">1.</span>
                <span><strong>Input Features (X):</strong> The characteristics we use to make predictions</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="font-bold">2.</span>
                <span><strong>Target Labels (y):</strong> The correct answers we want to predict</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="font-bold">3.</span>
                <span><strong>Learning Algorithm:</strong> Finds patterns between features and labels</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="font-bold">4.</span>
                <span><strong>Model:</strong> The learned function that makes predictions</span>
              </div>
            </div>
          </div>
          
          <div className="bg-white rounded p-4">
            <h4 className="font-semibold text-blue-900 mb-3">Prediction Process</h4>
            <div className="space-y-2 text-sm text-blue-800">
              <div className="flex items-start space-x-2">
                <span className="font-bold">1.</span>
                <span><strong>New Data:</strong> Unseen examples with only features (no labels)</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="font-bold">2.</span>
                <span><strong>Apply Model:</strong> Use the trained model on new features</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="font-bold">3.</span>
                <span><strong>Generate Predictions:</strong> Output the predicted labels</span>
              </div>
              <div className="flex items-start space-x-2">
                <span className="font-bold">4.</span>
                <span><strong>Confidence Scores:</strong> How certain the model is about predictions</span>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-6 bg-white rounded p-4">
          <h4 className="font-semibold text-blue-900 mb-3">Mathematical Foundation</h4>
          <p className="text-blue-800 text-sm mb-3">
            Supervised learning aims to find a function f that maps inputs X to outputs y:
          </p>
          <div className="bg-blue-100 rounded p-3 font-mono text-sm text-center">
            y = f(X) + ε
          </div>
          <p className="text-blue-700 text-xs mt-2">
            Where ε represents noise or irreducible error in the data
          </p>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-gray-100 rounded-lg p-1">
        <button
          onClick={() => setActiveTab('regression')}
          className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'regression' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-600'
          }`}
        >
          Regression
        </button>
        <button
          onClick={() => setActiveTab('classification')}
          className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'classification' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-600'
          }`}
        >
          Classification
        </button>
        <button
          onClick={() => setActiveTab('comparison')}
          className={`flex-1 px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'comparison' ? 'bg-white text-blue-600 shadow-sm' : 'text-gray-600'
          }`}
        >
          Comparison
        </button>
      </div>

      {/* Regression Content */}
      {activeTab === 'regression' && (
        <div className="space-y-6">
          <div className="bg-green-50 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-green-900 mb-3">Regression: Predicting Continuous Values</h3>
            <p className="text-green-800 mb-4">
              Regression predicts continuous numerical values. The goal is to find the best line (or curve) 
              that fits through the data points, minimizing the difference between predicted and actual values.
            </p>
            
            {/* Algorithm Selector */}
            <div className="mb-6">
              <h4 className="font-medium text-green-900 mb-3">Choose Algorithm to Explore:</h4>
              <div className="flex flex-wrap gap-2">
                {[
                  { id: 'linear', name: 'Linear Regression', complexity: 'Simple' },
                  { id: 'polynomial', name: 'Polynomial Regression', complexity: 'Medium' },
                  { id: 'ridge', name: 'Ridge Regression', complexity: 'Medium' },
                  { id: 'lasso', name: 'Lasso Regression', complexity: 'Medium' },
                  { id: 'random-forest', name: 'Random Forest', complexity: 'Complex' }
                ].map((algo) => (
                  <button
                    key={algo.id}
                    onClick={() => setSelectedAlgorithm(algo.id)}
                    className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                      selectedAlgorithm === algo.id
                        ? 'bg-green-600 text-white'
                        : 'bg-white text-green-700 hover:bg-green-100'
                    }`}
                  >
                    {algo.name}
                    <span className="ml-1 text-xs opacity-75">({algo.complexity})</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Algorithm Details */}
            {selectedAlgorithm === 'linear' && (
              <div className="bg-white rounded p-4 mb-4">
                <h4 className="font-semibold text-green-900 mb-3">Linear Regression</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h5 className="font-medium text-green-800 mb-2">Mathematical Formula:</h5>
                    <div className="bg-green-100 rounded p-3 font-mono text-sm mb-3">
                      y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
                    </div>
                    <div className="text-green-700 text-sm space-y-1">
                      <div>• β₀: Intercept (bias term)</div>
                      <div>• β₁, β₂, ..., βₙ: Coefficients (weights)</div>
                      <div>• x₁, x₂, ..., xₙ: Input features</div>
                      <div>• ε: Error term</div>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-medium text-green-800 mb-2">How it Works:</h5>
                    <div className="text-green-700 text-sm space-y-2">
                      <div><strong>1. Cost Function:</strong> Mean Squared Error (MSE)</div>
                      <div className="bg-green-100 rounded p-2 font-mono text-xs">
                        MSE = (1/n) Σ(yᵢ - ŷᵢ)²
                      </div>
                      <div><strong>2. Optimization:</strong> Minimize MSE using calculus</div>
                      <div><strong>3. Solution:</strong> Normal equation or gradient descent</div>
                      <div><strong>4. Result:</strong> Best-fit line through data points</div>
                    </div>
                  </div>
                </div>
                
                <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <h5 className="font-medium text-green-800 mb-2">Advantages:</h5>
                    <ul className="text-green-700 text-sm space-y-1">
                      <li>• Simple and interpretable</li>
                      <li>• Fast training and prediction</li>
                      <li>• No hyperparameters to tune</li>
                      <li>• Good baseline model</li>
                      <li>• Works well with linear relationships</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-medium text-green-800 mb-2">Disadvantages:</h5>
                    <ul className="text-green-700 text-sm space-y-1">
                      <li>• Assumes linear relationships</li>
                      <li>• Sensitive to outliers</li>
                      <li>• Can overfit with many features</li>
                      <li>• Requires feature scaling</li>
                      <li>• Poor with non-linear patterns</li>
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-medium text-green-800 mb-2">Best Use Cases:</h5>
                    <ul className="text-green-700 text-sm space-y-1">
                      <li>• Simple prediction problems</li>
                      <li>• When interpretability is key</li>
                      <li>• Linear relationships in data</li>
                      <li>• Small to medium datasets</li>
                      <li>• Baseline model comparison</li>
                    </ul>
                  </div>
                </div>
              </div>
            )}

            {selectedAlgorithm === 'polynomial' && (
              <div className="bg-white rounded p-4 mb-4">
                <h4 className="font-semibold text-green-900 mb-3">Polynomial Regression</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h5 className="font-medium text-green-800 mb-2">Mathematical Formula:</h5>
                    <div className="bg-green-100 rounded p-3 font-mono text-sm mb-3">
                      y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ
                    </div>
                    <div className="text-green-700 text-sm space-y-1">
                      <div>• Extends linear regression with polynomial terms</div>
                      <div>• Can capture non-linear relationships</div>
                      <div>• Degree determines complexity</div>
                      <div>• Still linear in parameters</div>
                    </div>
                  </div>
                  <div>
                    <h5 className="font-medium text-green-800 mb-2">Feature Engineering:</h5>
                    <div className="text-green-700 text-sm space-y-2">
                      <div><strong>Original features:</strong> [x₁, x₂]</div>
                      <div><strong>Degree 2 features:</strong> [x₁, x₂, x₁², x₁x₂, x₂²]</div>
                      <div><strong>Degree 3 features:</strong> [x₁, x₂, x₁², x₁x₂, x₂², x₁³, x₁²x₂, x₁x₂², x₂³]</div>
                      <div className="text-yellow-700"><strong>Warning:</strong> Features grow exponentially!</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Interactive House Price Prediction */}
            <div className="bg-gray-50 rounded-lg p-6">
              <h4 className="font-semibold text-gray-900 mb-3">Interactive Example: House Price Prediction</h4>
              <p className="text-gray-700 text-sm mb-4">
                Adjust the sliders to see how different features affect the predicted house price. 
                This demonstrates how regression models combine multiple inputs to make predictions.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        House Size: {houseSize.toLocaleString()} sq ft
                      </label>
                      <input 
                        type="range" 
                        min="500" 
                        max="5000" 
                        value={houseSize}
                        onChange={(e) => setHouseSize(parseInt(e.target.value))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>500 sq ft</span>
                        <span>5,000 sq ft</span>
                      </div>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Bedrooms: {bedrooms}
                      </label>
                      <input 
                        type="range" 
                        min="1" 
                        max="6" 
                        value={bedrooms}
                        onChange={(e) => setBedrooms(parseInt(e.target.value))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>1 bedroom</span>
                        <span>6 bedrooms</span>
                      </div>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Age: {houseAge} years
                      </label>
                      <input 
                        type="range" 
                        min="0" 
                        max="50" 
                        value={houseAge}
                        onChange={(e) => setHouseAge(parseInt(e.target.value))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>New</span>
                        <span>50 years</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-white rounded-lg p-6 border-2 border-dashed border-gray-300">
                  <div className="text-center">
                    <div className="text-4xl font-bold text-blue-600 mb-2">
                      ${predictedPrice.toLocaleString()}
                    </div>
                    <div className="text-sm text-gray-600 mb-4">Predicted Price</div>
                    
                    <div className="text-left space-y-2 text-xs text-gray-500">
                      <div className="font-medium mb-2">Calculation Breakdown:</div>
                      <div>Base Price: $50,000</div>
                      <div>Size ({houseSize} sq ft × $150): +${(houseSize * 150).toLocaleString()}</div>
                      <div>Bedrooms ({bedrooms} × $25,000): +${(bedrooms * 25000).toLocaleString()}</div>
                      <div>Age ({houseAge} years × -$2,000): -${(houseAge * 2000).toLocaleString()}</div>
                      <div className="border-t pt-2 font-medium">
                        Total: ${predictedPrice.toLocaleString()}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mt-6 bg-white rounded p-4">
                <h5 className="font-medium text-gray-900 mb-2">Model Insights:</h5>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <strong className="text-green-600">Positive Coefficients:</strong>
                    <div className="text-gray-600">Size and bedrooms increase price</div>
                  </div>
                  <div>
                    <strong className="text-red-600">Negative Coefficients:</strong>
                    <div className="text-gray-600">Age decreases price (depreciation)</div>
                  </div>
                  <div>
                    <strong className="text-blue-600">Feature Importance:</strong>
                    <div className="text-gray-600">Size has the largest impact</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Common Regression Metrics */}
            <div className="bg-white rounded p-4">
              <h4 className="font-semibold text-green-900 mb-3">Regression Evaluation Metrics</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h5 className="font-medium text-green-800 mb-2">Error-Based Metrics:</h5>
                  <div className="space-y-3 text-sm">
                    <div className="bg-green-50 rounded p-3">
                      <div className="font-medium">Mean Absolute Error (MAE)</div>
                      <div className="font-mono text-xs mt-1">MAE = (1/n) Σ|yᵢ - ŷᵢ|</div>
                      <div className="text-green-700 mt-1">Average absolute difference between predictions and actual values</div>
                    </div>
                    <div className="bg-green-50 rounded p-3">
                      <div className="font-medium">Mean Squared Error (MSE)</div>
                      <div className="font-mono text-xs mt-1">MSE = (1/n) Σ(yᵢ - ŷᵢ)²</div>
                      <div className="text-green-700 mt-1">Penalizes larger errors more heavily than smaller ones</div>
                    </div>
                    <div className="bg-green-50 rounded p-3">
                      <div className="font-medium">Root Mean Squared Error (RMSE)</div>
                      <div className="font-mono text-xs mt-1">RMSE = √MSE</div>
                      <div className="text-green-700 mt-1">Same units as target variable, easier to interpret</div>
                    </div>
                  </div>
                </div>
                <div>
                  <h5 className="font-medium text-green-800 mb-2">Goodness-of-Fit Metrics:</h5>
                  <div className="space-y-3 text-sm">
                    <div className="bg-green-50 rounded p-3">
                      <div className="font-medium">R-squared (R²)</div>
                      <div className="font-mono text-xs mt-1">R² = 1 - (SS_res / SS_tot)</div>
                      <div className="text-green-700 mt-1">Proportion of variance explained by the model (0-1)</div>
                    </div>
                    <div className="bg-green-50 rounded p-3">
                      <div className="font-medium">Adjusted R-squared</div>
                      <div className="font-mono text-xs mt-1">Adj R² = 1 - [(1-R²)(n-1)/(n-k-1)]</div>
                      <div className="text-green-700 mt-1">Penalizes adding irrelevant features</div>
                    </div>
                    <div className="bg-green-50 rounded p-3">
                      <div className="font-medium">Mean Absolute Percentage Error</div>
                      <div className="font-mono text-xs mt-1">MAPE = (100/n) Σ|((yᵢ-ŷᵢ)/yᵢ)|</div>
                      <div className="text-green-700 mt-1">Percentage error, useful for comparing across scales</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Classification Content */}
      {activeTab === 'classification' && (
        <div className="space-y-6">
          <div className="bg-purple-50 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-purple-900 mb-3">Classification: Predicting Categories</h3>
            <p className="text-purple-800 mb-4">
              Classification predicts discrete categories or classes. Instead of predicting a number, 
              we predict which category an example belongs to. The model learns decision boundaries 
              that separate different classes in the feature space.
            </p>
            
            {/* Types of Classification */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-white rounded p-4">
                <h4 className="font-semibold text-purple-900 mb-2">Binary Classification</h4>
                <div className="text-purple-800 text-sm space-y-1">
                  <div>• Two possible outcomes</div>
                  <div>• Examples: Spam/Not Spam, Fraud/Legitimate</div>
                  <div>• Output: 0 or 1, True or False</div>
                  <div>• Most common type</div>
                </div>
              </div>
              <div className="bg-white rounded p-4">
                <h4 className="font-semibold text-purple-900 mb-2">Multi-class Classification</h4>
                <div className="text-purple-800 text-sm space-y-1">
                  <div>• Multiple mutually exclusive classes</div>
                  <div>• Examples: Animal species, Product categories</div>
                  <div>• Output: One of many possible classes</div>
                  <div>• Each example belongs to exactly one class</div>
                </div>
              </div>
              <div className="bg-white rounded p-4">
                <h4 className="font-semibold text-purple-900 mb-2">Multi-label Classification</h4>
                <div className="text-purple-800 text-sm space-y-1">
                  <div>• Multiple non-exclusive labels</div>
                  <div>• Examples: Movie genres, Medical conditions</div>
                  <div>• Output: Set of applicable labels</div>
                  <div>• Each example can have multiple labels</div>
                </div>
              </div>
            </div>

            {/* Logistic Regression Deep Dive */}
            <div className="bg-white rounded p-4 mb-4">
              <h4 className="font-semibold text-purple-900 mb-3">Logistic Regression: The Foundation</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h5 className="font-medium text-purple-800 mb-2">Why Not Linear Regression?</h5>
                  <div className="text-purple-700 text-sm space-y-2">
                    <div>• Linear regression can predict values outside [0,1]</div>
                    <div>• We need probabilities that sum to 1</div>
                    <div>• Decision boundaries should be smooth</div>
                    <div>• Need to handle non-linear relationships</div>
                  </div>
                  
                  <h5 className="font-medium text-purple-800 mb-2 mt-4">The Sigmoid Function</h5>
                  <div className="bg-purple-100 rounded p-3 font-mono text-sm mb-2">
                    σ(z) = 1 / (1 + e^(-z))
                  </div>
                  <div className="text-purple-700 text-sm">
                    Maps any real number to a value between 0 and 1
                  </div>
                </div>
                <div>
                  <h5 className="font-medium text-purple-800 mb-2">Complete Formula</h5>
                  <div className="space-y-2 text-sm">
                    <div className="bg-purple-100 rounded p-2 font-mono">
                      z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
                    </div>
                    <div className="bg-purple-100 rounded p-2 font-mono">
                      P(y=1|x) = 1 / (1 + e^(-z))
                    </div>
                    <div className="text-purple-700">
                      <div>• z: Linear combination of features</div>
                      <div>• P(y=1|x): Probability of positive class</div>
                      <div>• Decision threshold: typically 0.5</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Interactive Email Spam Detection */}
            <div className="bg-gray-50 rounded-lg p-6">
              <h4 className="font-semibold text-gray-900 mb-3">Interactive Example: Email Spam Detection</h4>
              <p className="text-gray-700 text-sm mb-4">
                Enter email content below to see how a classification model would analyze it. 
                The model looks for patterns in text that indicate spam vs. legitimate emails.
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Email Content:
                  </label>
                  <textarea 
                    className="w-full h-32 p-3 border border-gray-300 rounded-lg resize-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                    placeholder="Enter email content to classify..."
                    value={emailContent}
                    onChange={(e) => setEmailContent(e.target.value)}
                  />
                  <button className="mt-3 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
                    Analyze Email
                  </button>
                </div>
                
                <div className="bg-white rounded-lg p-4 border-2 border-dashed border-gray-300">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-red-600 mb-2">🚨 SPAM</div>
                    <div className="text-sm text-gray-600 mb-3">Confidence: 94.7%</div>
                    
                    <div className="text-left text-xs text-gray-500 space-y-2">
                      <div><strong>Detected Features:</strong></div>
                      <div className="bg-red-50 rounded p-2">
                        <div>• Money-related keywords: "won", "$1,000,000"</div>
                        <div>• Urgency indicators: "URGENT", "NOW", "expires"</div>
                        <div>• Suspicious phrases: "Click here", "claim your prize"</div>
                        <div>• Excessive capitalization and exclamation marks</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Comparison Tab */}
      {activeTab === 'comparison' && (
        <div className="space-y-6">
          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">Regression vs Classification: Complete Comparison</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
              <div className="bg-green-50 rounded-lg p-4">
                <h4 className="font-semibold text-green-900 mb-3">🔢 Regression</h4>
                <div className="space-y-3 text-sm">
                  <div>
                    <strong className="text-green-800">Output Type:</strong>
                    <div className="text-green-700">Continuous numerical values</div>
                  </div>
                  <div>
                    <strong className="text-green-800">Examples:</strong>
                    <div className="text-green-700">House prices, stock prices, temperature, sales revenue</div>
                  </div>
                  <div>
                    <strong className="text-green-800">Goal:</strong>
                    <div className="text-green-700">Minimize prediction error (MSE, MAE)</div>
                  </div>
                  <div>
                    <strong className="text-green-800">Output Range:</strong>
                    <div className="text-green-700">Unlimited (can be any real number)</div>
                  </div>
                </div>
              </div>
              
              <div className="bg-purple-50 rounded-lg p-4">
                <h4 className="font-semibold text-purple-900 mb-3">🏷️ Classification</h4>
                <div className="space-y-3 text-sm">
                  <div>
                    <strong className="text-purple-800">Output Type:</strong>
                    <div className="text-purple-700">Discrete categories or classes</div>
                  </div>
                  <div>
                    <strong className="text-purple-800">Examples:</strong>
                    <div className="text-purple-700">Spam detection, image recognition, medical diagnosis</div>
                  </div>
                  <div>
                    <strong className="text-purple-800">Goal:</strong>
                    <div className="text-purple-700">Maximize classification accuracy</div>
                  </div>
                  <div>
                    <strong className="text-purple-800">Output Range:</strong>
                    <div className="text-purple-700">Limited to predefined classes</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="flex justify-between items-center pt-6 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('introduction')}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
        >
          ← Previous: Introduction
        </button>
        <div className="text-sm text-gray-500">Lesson 2 of 6 • Estimated time: 60 minutes</div>
        <button 
          onClick={() => setActiveLesson('unsupervised')}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Next: Unsupervised Learning →
        </button>
      </div>
    </div>
  )
}

function UnsupervisedLearningLesson({ setActiveLesson }: LessonProps) {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Unsupervised Learning</h2>
        <p className="text-lg text-gray-600 mb-6">
          Discover hidden patterns and structures in data without labeled examples.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-blue-50 rounded-lg p-6">
          <h3 className="text-xl font-semibold text-blue-900 mb-3">K-Means Clustering</h3>
          <p className="text-blue-800 mb-4">
            Groups similar data points into k clusters by minimizing within-cluster variance.
          </p>
          <div className="bg-white rounded p-4 font-mono text-sm mb-4">
            1. Choose k (number of clusters)<br/>
            2. Initialize cluster centers<br/>
            3. Assign points to nearest center<br/>
            4. Update centers<br/>
            5. Repeat until convergence
          </div>
          <div className="text-blue-800 text-sm">
            <strong>Use cases:</strong> Customer segmentation, image segmentation, market research
          </div>
        </div>

        <div className="bg-green-50 rounded-lg p-6">
          <h3 className="text-xl font-semibold text-green-900 mb-3">Principal Component Analysis</h3>
          <p className="text-green-800 mb-4">
            Reduces dimensionality while preserving the most important information.
          </p>
          <div className="bg-white rounded p-4 font-mono text-sm mb-4">
            1. Standardize the data<br/>
            2. Compute covariance matrix<br/>
            3. Find eigenvalues/eigenvectors<br/>
            4. Select top k components<br/>
            5. Transform data
          </div>
          <div className="text-green-800 text-sm">
            <strong>Use cases:</strong> Data visualization, noise reduction, feature extraction
          </div>
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Interactive Clustering Demo</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <div className="bg-white rounded-lg p-4 h-64 border-2 border-dashed border-gray-300 flex items-center justify-center">
              <div className="text-center text-gray-500">
                <div className="text-4xl mb-2">📊</div>
                <div>Interactive clustering visualization</div>
                <div className="text-sm">(Click to generate random data points)</div>
              </div>
            </div>
            <div className="mt-4 flex space-x-2">
              <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors">
                Generate Data
              </button>
              <button className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors">
                Run K-Means
              </button>
              <select className="px-3 py-2 border border-gray-300 rounded">
                <option>k = 2</option>
                <option>k = 3</option>
                <option>k = 4</option>
                <option>k = 5</option>
              </select>
            </div>
          </div>
          <div>
            <h4 className="font-semibold text-gray-900 mb-3">Algorithm Steps:</h4>
            <div className="space-y-3">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-semibold text-sm">1</div>
                <div className="text-sm text-gray-700">Initialize k cluster centers randomly</div>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-semibold text-sm">2</div>
                <div className="text-sm text-gray-700">Assign each point to nearest center</div>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-semibold text-sm">3</div>
                <div className="text-sm text-gray-700">Update centers to cluster means</div>
              </div>
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-semibold text-sm">4</div>
                <div className="text-sm text-gray-700">Repeat until centers stop moving</div>
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-yellow-50 rounded-lg">
              <h5 className="font-medium text-yellow-900 mb-2">💡 Pro Tip</h5>
              <p className="text-yellow-800 text-sm">
                Use the "elbow method" to find the optimal number of clusters by plotting 
                the within-cluster sum of squares against k.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-between items-center pt-6 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('supervised')}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
        >
          ← Previous: Supervised Learning
        </button>
        <div className="text-sm text-gray-500">Lesson 3 of 6</div>
        <button 
          onClick={() => setActiveLesson('evaluation')}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Next: Model Evaluation →
        </button>
      </div>
    </div>
  )
}

function ModelEvaluationLesson({ setActiveLesson }: LessonProps) {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Model Evaluation</h2>
        <p className="text-lg text-gray-600 mb-6">
          Learn how to assess and improve your machine learning models.
        </p>
      </div>

      <div className="bg-red-50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-red-900 mb-3">The Bias-Variance Tradeoff</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg p-4">
            <h4 className="font-semibold text-red-900 mb-2">High Bias (Underfitting)</h4>
            <ul className="text-red-800 text-sm space-y-1">
              <li>• Too simple model</li>
              <li>• Poor performance on training data</li>
              <li>• High training error</li>
              <li>• High validation error</li>
            </ul>
          </div>
          <div className="bg-white rounded-lg p-4">
            <h4 className="font-semibold text-green-900 mb-2">Balanced Model</h4>
            <ul className="text-green-800 text-sm space-y-1">
              <li>• Appropriate complexity</li>
              <li>• Good generalization</li>
              <li>• Low training error</li>
              <li>• Low validation error</li>
            </ul>
          </div>
          <div className="bg-white rounded-lg p-4">
            <h4 className="font-semibold text-purple-900 mb-2">High Variance (Overfitting)</h4>
            <ul className="text-purple-800 text-sm space-y-1">
              <li>• Too complex model</li>
              <li>• Memorizes training data</li>
              <li>• Low training error</li>
              <li>• High validation error</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-blue-50 rounded-lg p-6">
          <h3 className="text-xl font-semibold text-blue-900 mb-3">Classification Metrics</h3>
          <div className="space-y-4">
            <div className="bg-white rounded p-4">
              <h4 className="font-medium text-blue-900 mb-2">Confusion Matrix</h4>
              <div className="grid grid-cols-3 gap-1 text-xs">
                <div></div>
                <div className="text-center font-medium">Predicted</div>
                <div></div>
                <div className="font-medium">Actual</div>
                <div className="bg-green-100 p-2 text-center">TP: 85</div>
                <div className="bg-red-100 p-2 text-center">FN: 15</div>
                <div></div>
                <div className="bg-red-100 p-2 text-center">FP: 10</div>
                <div className="bg-green-100 p-2 text-center">TN: 90</div>
              </div>
            </div>
            <div className="bg-white rounded p-4 space-y-2">
              <div className="flex justify-between">
                <span className="text-blue-800">Accuracy:</span>
                <span className="font-mono">87.5%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-800">Precision:</span>
                <span className="font-mono">89.5%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-800">Recall:</span>
                <span className="font-mono">85.0%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-blue-800">F1-Score:</span>
                <span className="font-mono">87.2%</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-green-50 rounded-lg p-6">
          <h3 className="text-xl font-semibold text-green-900 mb-3">Cross-Validation</h3>
          <p className="text-green-800 mb-4">
            Robust method to evaluate model performance using multiple train/test splits.
          </p>
          <div className="bg-white rounded p-4 mb-4">
            <h4 className="font-medium text-green-900 mb-2">K-Fold Cross-Validation</h4>
            <div className="space-y-2">
              <div className="flex space-x-1">
                <div className="flex-1 bg-blue-200 h-6 rounded flex items-center justify-center text-xs">Test</div>
                <div className="flex-1 bg-gray-200 h-6 rounded flex items-center justify-center text-xs">Train</div>
                <div className="flex-1 bg-gray-200 h-6 rounded flex items-center justify-center text-xs">Train</div>
                <div className="flex-1 bg-gray-200 h-6 rounded flex items-center justify-center text-xs">Train</div>
                <div className="flex-1 bg-gray-200 h-6 rounded flex items-center justify-center text-xs">Train</div>
              </div>
              <div className="flex space-x-1">
                <div className="flex-1 bg-gray-200 h-6 rounded flex items-center justify-center text-xs">Train</div>
                <div className="flex-1 bg-blue-200 h-6 rounded flex items-center justify-center text-xs">Test</div>
                <div className="flex-1 bg-gray-200 h-6 rounded flex items-center justify-center text-xs">Train</div>
                <div className="flex-1 bg-gray-200 h-6 rounded flex items-center justify-center text-xs">Train</div>
                <div className="flex-1 bg-gray-200 h-6 rounded flex items-center justify-center text-xs">Train</div>
              </div>
              <div className="text-center text-xs text-green-700">... and so on for k=5 folds</div>
            </div>
          </div>
          <div className="bg-white rounded p-4">
            <div className="text-sm text-green-800">
              <div className="font-medium mb-2">Benefits:</div>
              <ul className="space-y-1">
                <li>• More reliable performance estimate</li>
                <li>• Uses all data for training and testing</li>
                <li>• Reduces variance in evaluation</li>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-between items-center pt-6 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('unsupervised')}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
        >
          ← Previous: Unsupervised Learning
        </button>
        <div className="text-sm text-gray-500">Lesson 4 of 6</div>
        <button 
          onClick={() => setActiveLesson('algorithms')}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Next: Key Algorithms →
        </button>
      </div>
    </div>
  )
}

function AlgorithmsLesson({ setActiveLesson }: LessonProps) {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Key ML Algorithms</h2>
        <p className="text-lg text-gray-600 mb-6">
          Master the most important machine learning algorithms and when to use them.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
          <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
            <span className="text-2xl">🌳</span>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Decision Trees</h3>
          <p className="text-gray-600 text-sm mb-3">
            Tree-like model that makes decisions by splitting data based on feature values.
          </p>
          <div className="text-xs text-gray-500 space-y-1">
            <div><strong>Pros:</strong> Interpretable, handles mixed data types</div>
            <div><strong>Cons:</strong> Prone to overfitting</div>
            <div><strong>Use for:</strong> Classification, regression</div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
          <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
            <span className="text-2xl">🌲</span>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Random Forest</h3>
          <p className="text-gray-600 text-sm mb-3">
            Ensemble of decision trees that votes on the final prediction.
          </p>
          <div className="text-xs text-gray-500 space-y-1">
            <div><strong>Pros:</strong> Reduces overfitting, feature importance</div>
            <div><strong>Cons:</strong> Less interpretable than single tree</div>
            <div><strong>Use for:</strong> Most tabular data problems</div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
          <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
            <span className="text-2xl">🎯</span>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Support Vector Machine</h3>
          <p className="text-gray-600 text-sm mb-3">
            Finds optimal boundary between classes by maximizing margin.
          </p>
          <div className="text-xs text-gray-500 space-y-1">
            <div><strong>Pros:</strong> Works well with high dimensions</div>
            <div><strong>Cons:</strong> Slow on large datasets</div>
            <div><strong>Use for:</strong> Text classification, image recognition</div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
          <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center mb-4">
            <span className="text-2xl">🔍</span>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">K-Nearest Neighbors</h3>
          <p className="text-gray-600 text-sm mb-3">
            Classifies based on the majority class of k nearest neighbors.
          </p>
          <div className="text-xs text-gray-500 space-y-1">
            <div><strong>Pros:</strong> Simple, no training required</div>
            <div><strong>Cons:</strong> Slow prediction, sensitive to scale</div>
            <div><strong>Use for:</strong> Recommendation systems</div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
          <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center mb-4">
            <span className="text-2xl">📈</span>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Gradient Boosting</h3>
          <p className="text-gray-600 text-sm mb-3">
            Sequentially builds models that correct previous model's errors.
          </p>
          <div className="text-xs text-gray-500 space-y-1">
            <div><strong>Pros:</strong> High accuracy, handles missing values</div>
            <div><strong>Cons:</strong> Can overfit, requires tuning</div>
            <div><strong>Use for:</strong> Competitions, structured data</div>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-lg transition-shadow">
          <div className="w-12 h-12 bg-indigo-100 rounded-lg flex items-center justify-center mb-4">
            <span className="text-2xl">🧮</span>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Naive Bayes</h3>
          <p className="text-gray-600 text-sm mb-3">
            Probabilistic classifier based on Bayes' theorem with independence assumption.
          </p>
          <div className="text-xs text-gray-500 space-y-1">
            <div><strong>Pros:</strong> Fast, works with small datasets</div>
            <div><strong>Cons:</strong> Strong independence assumption</div>
            <div><strong>Use for:</strong> Text classification, spam filtering</div>
          </div>
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Algorithm Selection Guide</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-900 mb-3">Choose based on data size:</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Small dataset (&lt;1K samples):</span>
                <span className="font-medium">Naive Bayes, KNN</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Medium dataset (1K-100K):</span>
                <span className="font-medium">Random Forest, SVM</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Large dataset (&gt;100K):</span>
                <span className="font-medium">Gradient Boosting, Neural Networks</span>
              </div>
            </div>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 mb-3">Choose based on problem type:</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Interpretability needed:</span>
                <span className="font-medium">Decision Trees, Linear Models</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">High accuracy needed:</span>
                <span className="font-medium">Ensemble methods, Deep Learning</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Fast prediction needed:</span>
                <span className="font-medium">Linear Models, Naive Bayes</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-between items-center pt-6 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('evaluation')}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
        >
          ← Previous: Model Evaluation
        </button>
        <div className="text-sm text-gray-500">Lesson 5 of 6</div>
        <button 
          onClick={() => setActiveLesson('projects')}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Next: Hands-on Projects →
        </button>
      </div>
    </div>
  )
}

function ProjectsLesson({ setActiveLesson }: LessonProps) {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold text-gray-900 mb-4">Hands-on Projects</h2>
        <p className="text-lg text-gray-600 mb-6">
          Apply your knowledge with real-world projects and build your portfolio.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center">
              <span className="text-white text-xl">🏠</span>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-blue-900">House Price Prediction</h3>
              <div className="text-blue-700 text-sm">Beginner • 2-3 hours</div>
            </div>
          </div>
          <p className="text-blue-800 mb-4">
            Predict house prices using features like size, location, and amenities. 
            Learn regression techniques and feature engineering.
          </p>
          <div className="space-y-2 mb-4">
            <div className="text-blue-900 text-sm font-medium">You'll learn:</div>
            <ul className="text-blue-800 text-sm space-y-1">
              <li>• Data preprocessing and cleaning</li>
              <li>• Feature selection and engineering</li>
              <li>• Linear and polynomial regression</li>
              <li>• Model evaluation and validation</li>
            </ul>
          </div>
          <button className="w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            Start Project
          </button>
        </div>

        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-green-500 rounded-lg flex items-center justify-center">
              <span className="text-white text-xl">📧</span>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-green-900">Email Spam Detection</h3>
              <div className="text-green-700 text-sm">Intermediate • 3-4 hours</div>
            </div>
          </div>
          <p className="text-green-800 mb-4">
            Build a classifier to detect spam emails using natural language processing 
            and machine learning techniques.
          </p>
          <div className="space-y-2 mb-4">
            <div className="text-green-900 text-sm font-medium">You'll learn:</div>
            <ul className="text-green-800 text-sm space-y-1">
              <li>• Text preprocessing and tokenization</li>
              <li>• TF-IDF vectorization</li>
              <li>• Classification algorithms</li>
              <li>• Performance metrics for imbalanced data</li>
            </ul>
          </div>
          <button className="w-full px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors">
            Start Project
          </button>
        </div>

        <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-purple-500 rounded-lg flex items-center justify-center">
              <span className="text-white text-xl">👥</span>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-purple-900">Customer Segmentation</h3>
              <div className="text-purple-700 text-sm">Intermediate • 4-5 hours</div>
            </div>
          </div>
          <p className="text-purple-800 mb-4">
            Segment customers based on purchasing behavior using clustering algorithms 
            to improve marketing strategies.
          </p>
          <div className="space-y-2 mb-4">
            <div className="text-purple-900 text-sm font-medium">You'll learn:</div>
            <ul className="text-purple-800 text-sm space-y-1">
              <li>• Exploratory data analysis</li>
              <li>• K-means and hierarchical clustering</li>
              <li>• Dimensionality reduction with PCA</li>
              <li>• Business insights from clusters</li>
            </ul>
          </div>
          <button className="w-full px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
            Start Project
          </button>
        </div>

        <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="w-12 h-12 bg-red-500 rounded-lg flex items-center justify-center">
              <span className="text-white text-xl">📈</span>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-red-900">Stock Price Prediction</h3>
              <div className="text-red-700 text-sm">Advanced • 6-8 hours</div>
            </div>
          </div>
          <p className="text-red-800 mb-4">
            Predict stock prices using time series analysis and machine learning. 
            Learn to work with financial data and time-based features.
          </p>
          <div className="space-y-2 mb-4">
            <div className="text-red-900 text-sm font-medium">You'll learn:</div>
            <ul className="text-red-800 text-sm space-y-1">
              <li>• Time series data preprocessing</li>
              <li>• Feature engineering for temporal data</li>
              <li>• LSTM neural networks</li>
              <li>• Financial metrics and evaluation</li>
            </ul>
          </div>
          <button className="w-full px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors">
            Start Project
          </button>
        </div>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Project Guidelines</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-900 mb-3">📋 What's Included:</h4>
            <ul className="text-gray-700 text-sm space-y-2">
              <li>• Step-by-step instructions</li>
              <li>• Starter code and datasets</li>
              <li>• Video walkthroughs</li>
              <li>• Solution notebooks</li>
              <li>• Community discussion forums</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-gray-900 mb-3">🎯 Learning Outcomes:</h4>
            <ul className="text-gray-700 text-sm space-y-2">
              <li>• Hands-on coding experience</li>
              <li>• Real-world problem solving</li>
              <li>• Portfolio projects</li>
              <li>• Industry best practices</li>
              <li>• Peer code reviews</li>
            </ul>
          </div>
        </div>
      </div>

      <div className="flex justify-between items-center pt-6 border-t border-gray-200">
        <button 
          onClick={() => setActiveLesson('algorithms')}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
        >
          ← Previous: Key Algorithms
        </button>
        <div className="text-sm text-gray-500">Lesson 6 of 6</div>
        <button className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
          Complete Course 🎉
        </button>
      </div>
    </div>
  )
} 