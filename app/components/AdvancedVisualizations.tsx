'use client'

import { useEffect, useRef, useState } from 'react'

export default function AdvancedVisualizations() {
  const [activeViz, setActiveViz] = useState('gradient-descent')

  const visualizations = [
    { id: 'gradient-descent', title: 'Gradient Descent', icon: '⛰️' },
    { id: 'neural-training', title: 'Neural Network Training', icon: '🧠' },
    { id: 'clustering', title: 'K-Means Clustering', icon: '🎯' },
    { id: 'decision-boundary', title: 'Decision Boundaries', icon: '📊' },
    { id: 'backprop', title: 'Backpropagation', icon: '🔄' },
    { id: 'attention', title: 'Attention Mechanism', icon: '👁️' }
  ]

  return (
    <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Advanced Interactive Visualizations</h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Explore complex machine learning concepts through interactive, real-time visualizations
          </p>
        </div>

        {/* Visualization selector */}
        <div className="flex flex-wrap justify-center gap-3 mb-8">
          {visualizations.map((viz) => (
            <button
              key={viz.id}
              onClick={() => setActiveViz(viz.id)}
              className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2 ${
                activeViz === viz.id
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-white text-gray-700 border border-gray-300 hover:border-blue-400 hover:text-blue-600'
              }`}
            >
              <span className="text-lg">{viz.icon}</span>
              <span className="text-sm">{viz.title}</span>
            </button>
          ))}
        </div>

        {/* Visualization content */}
        <div className="bg-white rounded-xl shadow-lg p-8">
          {activeViz === 'gradient-descent' && <GradientDescentViz />}
          {activeViz === 'neural-training' && <NeuralTrainingViz />}
          {activeViz === 'clustering' && <ClusteringViz />}
          {activeViz === 'decision-boundary' && <DecisionBoundaryViz />}
          {activeViz === 'backprop' && <BackpropViz />}
          {activeViz === 'attention' && <AttentionViz />}
        </div>
      </div>
    </section>
  )
}

function GradientDescentViz() {
  const [learningRate, setLearningRate] = useState(0.1)
  const [isRunning, setIsRunning] = useState(false)
  const [currentPoint, setCurrentPoint] = useState({ x: 8, y: 0 })
  const [path, setPath] = useState([{ x: 8, y: 0 }])
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  // Simple quadratic function: f(x) = (x-5)^2 + 2
  const func = (x: number) => Math.pow(x - 5, 2) + 2
  const derivative = (x: number) => 2 * (x - 5)

  const step = () => {
    setCurrentPoint(prev => {
      const grad = derivative(prev.x)
      const newX = prev.x - learningRate * grad
      const newY = func(newX)
      const newPoint = { x: newX, y: newY }
      
      setPath(prevPath => [...prevPath, newPoint])
      
      // Stop if we're close to minimum or diverging
      if (Math.abs(grad) < 0.01 || Math.abs(newX) > 20) {
        setIsRunning(false)
        if (intervalRef.current) clearInterval(intervalRef.current)
      }
      
      return newPoint
    })
  }

  const startOptimization = () => {
    setIsRunning(true)
    intervalRef.current = setInterval(step, 500)
  }

  const reset = () => {
    setIsRunning(false)
    if (intervalRef.current) clearInterval(intervalRef.current)
    setCurrentPoint({ x: 8, y: func(8) })
    setPath([{ x: 8, y: func(8) }])
  }

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-2xl font-bold text-gray-900 mb-4">Gradient Descent Optimization</h3>
        <p className="text-gray-600 mb-6">
          Watch how gradient descent finds the minimum of a function by following the negative gradient.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <div className="bg-gray-50 rounded-lg p-4 h-80 flex items-center justify-center">
            <svg width="400" height="300" viewBox="0 0 400 300">
              {/* Function curve */}
              <path
                d={Array.from({ length: 100 }, (_, i) => {
                  const x = i * 0.2
                  const y = func(x)
                  const screenX = x * 20 + 50
                  const screenY = 250 - y * 20
                  return `${i === 0 ? 'M' : 'L'} ${screenX} ${screenY}`
                }).join(' ')}
                stroke="#3b82f6"
                strokeWidth="3"
                fill="none"
              />
              
              {/* Path taken by gradient descent */}
              {path.length > 1 && (
                <path
                  d={path.map((point, i) => {
                    const screenX = point.x * 20 + 50
                    const screenY = 250 - point.y * 20
                    return `${i === 0 ? 'M' : 'L'} ${screenX} ${screenY}`
                  }).join(' ')}
                  stroke="#ef4444"
                  strokeWidth="2"
                  fill="none"
                  strokeDasharray="5,5"
                />
              )}
              
              {/* Current point */}
              <circle
                cx={currentPoint.x * 20 + 50}
                cy={250 - currentPoint.y * 20}
                r="6"
                fill="#ef4444"
                className="animate-pulse"
              />
              
              {/* Minimum point */}
              <circle
                cx={5 * 20 + 50}
                cy={250 - 2 * 20}
                r="4"
                fill="#10b981"
              />
              
              {/* Axes */}
              <line x1="50" y1="250" x2="350" y2="250" stroke="#6b7280" strokeWidth="1" />
              <line x1="50" y1="50" x2="50" y2="250" stroke="#6b7280" strokeWidth="1" />
              
              {/* Labels */}
              <text x="200" y="280" textAnchor="middle" className="text-xs fill-gray-600">x</text>
              <text x="30" y="150" textAnchor="middle" className="text-xs fill-gray-600">f(x)</text>
            </svg>
          </div>
        </div>
        
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Learning Rate: {learningRate.toFixed(2)}
            </label>
            <input
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
              className="w-full"
              disabled={isRunning}
            />
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-2">Current Status:</h4>
            <div className="space-y-1 text-sm">
              <div>Position: x = {currentPoint.x.toFixed(3)}</div>
              <div>Function value: f(x) = {currentPoint.y.toFixed(3)}</div>
              <div>Gradient: f'(x) = {derivative(currentPoint.x).toFixed(3)}</div>
              <div>Steps taken: {path.length - 1}</div>
            </div>
          </div>
          
          <div className="flex space-x-3">
            <button
              onClick={startOptimization}
              disabled={isRunning}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isRunning ? 'Running...' : 'Start Optimization'}
            </button>
            <button
              onClick={reset}
              className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
            >
              Reset
            </button>
          </div>
          
          <div className="bg-yellow-50 rounded-lg p-4">
            <h5 className="font-medium text-yellow-900 mb-2">💡 Try This:</h5>
            <ul className="text-yellow-800 text-sm space-y-1">
              <li>• Increase learning rate - see what happens!</li>
              <li>• Very small learning rate = slow convergence</li>
              <li>• Very large learning rate = overshooting</li>
              <li>• Green dot shows the true minimum</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

function NeuralTrainingViz() {
  const [epoch, setEpoch] = useState(0)
  const [isTraining, setIsTraining] = useState(false)
  const [loss, setLoss] = useState(2.5)
  const [accuracy, setAccuracy] = useState(0.1)
  const [lossHistory, setLossHistory] = useState([2.5])
  const [accHistory, setAccHistory] = useState([0.1])

  const trainStep = () => {
    setEpoch(prev => prev + 1)
    setLoss(prev => Math.max(0.01, prev * (0.95 + Math.random() * 0.1)))
    setAccuracy(prev => Math.min(0.99, prev + (Math.random() * 0.05)))
    
    setLossHistory(prev => [...prev.slice(-49), loss])
    setAccHistory(prev => [...prev.slice(-49), accuracy])
  }

  const startTraining = () => {
    setIsTraining(true)
    const interval = setInterval(() => {
      trainStep()
    }, 200)
    
    setTimeout(() => {
      setIsTraining(false)
      clearInterval(interval)
    }, 10000)
  }

  const reset = () => {
    setEpoch(0)
    setLoss(2.5)
    setAccuracy(0.1)
    setLossHistory([2.5])
    setAccHistory([0.1])
    setIsTraining(false)
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-2xl font-bold text-gray-900 mb-4">Neural Network Training</h3>
        <p className="text-gray-600 mb-6">
          Observe how loss decreases and accuracy increases during neural network training.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <div className="bg-gray-50 rounded-lg p-4 h-80">
            <h4 className="font-semibold text-gray-900 mb-4">Training Metrics</h4>
            <svg width="100%" height="250" viewBox="0 0 400 250">
              {/* Loss curve */}
              <path
                d={lossHistory.map((val, i) => {
                  const x = (i / Math.max(lossHistory.length - 1, 1)) * 350 + 25
                  const y = 225 - (val / 3) * 180
                  return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
                }).join(' ')}
                stroke="#ef4444"
                strokeWidth="2"
                fill="none"
              />
              
              {/* Accuracy curve */}
              <path
                d={accHistory.map((val, i) => {
                  const x = (i / Math.max(accHistory.length - 1, 1)) * 350 + 25
                  const y = 225 - val * 180
                  return `${i === 0 ? 'M' : 'L'} ${x} ${y}`
                }).join(' ')}
                stroke="#10b981"
                strokeWidth="2"
                fill="none"
              />
              
              {/* Axes */}
              <line x1="25" y1="225" x2="375" y2="225" stroke="#6b7280" strokeWidth="1" />
              <line x1="25" y1="45" x2="25" y2="225" stroke="#6b7280" strokeWidth="1" />
              
              {/* Legend */}
              <line x1="300" y1="60" x2="320" y2="60" stroke="#ef4444" strokeWidth="2" />
              <text x="325" y="65" className="text-xs fill-gray-600">Loss</text>
              <line x1="300" y1="80" x2="320" y2="80" stroke="#10b981" strokeWidth="2" />
              <text x="325" y="85" className="text-xs fill-gray-600">Accuracy</text>
            </svg>
          </div>
        </div>
        
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-red-50 rounded-lg p-4">
              <h4 className="font-semibold text-red-900 mb-1">Loss</h4>
              <div className="text-2xl font-bold text-red-600">{loss.toFixed(3)}</div>
            </div>
            <div className="bg-green-50 rounded-lg p-4">
              <h4 className="font-semibold text-green-900 mb-1">Accuracy</h4>
              <div className="text-2xl font-bold text-green-600">{(accuracy * 100).toFixed(1)}%</div>
            </div>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-2">Training Progress:</h4>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Epoch: {epoch}</span>
                <span>Status: {isTraining ? 'Training...' : 'Stopped'}</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(epoch / 50 * 100, 100)}%` }}
                ></div>
              </div>
            </div>
          </div>
          
          <div className="flex space-x-3">
            <button
              onClick={startTraining}
              disabled={isTraining}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {isTraining ? 'Training...' : 'Start Training'}
            </button>
            <button
              onClick={reset}
              className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
            >
              Reset
            </button>
          </div>
          
          <div className="bg-blue-50 rounded-lg p-4">
            <h5 className="font-medium text-blue-900 mb-2">📈 What's Happening:</h5>
            <ul className="text-blue-800 text-sm space-y-1">
              <li>• Loss decreases as model learns patterns</li>
              <li>• Accuracy increases with better predictions</li>
              <li>• Training curves show learning progress</li>
              <li>• Real training has more fluctuations</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

function ClusteringViz() {
  const [k, setK] = useState(3)
  const [points, setPoints] = useState<Array<{x: number, y: number, cluster: number}>>([])
  const [centroids, setCentroids] = useState<Array<{x: number, y: number}>>([])
  const [isRunning, setIsRunning] = useState(false)

  const colors = ['#ef4444', '#10b981', '#3b82f6', '#f59e0b', '#8b5cf6']

  const generateData = () => {
    const newPoints = []
    // Generate 3 clusters of points
    for (let cluster = 0; cluster < 3; cluster++) {
      const centerX = 100 + cluster * 120 + Math.random() * 40
      const centerY = 100 + Math.random() * 80
      
      for (let i = 0; i < 15; i++) {
        newPoints.push({
          x: centerX + (Math.random() - 0.5) * 60,
          y: centerY + (Math.random() - 0.5) * 60,
          cluster: -1
        })
      }
    }
    setPoints(newPoints)
    
    // Initialize random centroids
    const newCentroids = []
    for (let i = 0; i < k; i++) {
      newCentroids.push({
        x: Math.random() * 300 + 50,
        y: Math.random() * 150 + 50
      })
    }
    setCentroids(newCentroids)
  }

  const runKMeans = () => {
    setIsRunning(true)
    
    const iterations = 10
    let currentPoints = [...points]
    let currentCentroids = [...centroids]
    
    for (let iter = 0; iter < iterations; iter++) {
      setTimeout(() => {
        // Assign points to nearest centroid
        currentPoints = currentPoints.map(point => {
          let minDist = Infinity
          let nearestCluster = 0
          
          currentCentroids.forEach((centroid, i) => {
            const dist = Math.sqrt(
              Math.pow(point.x - centroid.x, 2) + Math.pow(point.y - centroid.y, 2)
            )
            if (dist < minDist) {
              minDist = dist
              nearestCluster = i
            }
          })
          
          return { ...point, cluster: nearestCluster }
        })
        
        // Update centroids
        currentCentroids = currentCentroids.map((_, i) => {
          const clusterPoints = currentPoints.filter(p => p.cluster === i)
          if (clusterPoints.length === 0) return currentCentroids[i]
          
          const avgX = clusterPoints.reduce((sum, p) => sum + p.x, 0) / clusterPoints.length
          const avgY = clusterPoints.reduce((sum, p) => sum + p.y, 0) / clusterPoints.length
          
          return { x: avgX, y: avgY }
        })
        
        setPoints([...currentPoints])
        setCentroids([...currentCentroids])
        
        if (iter === iterations - 1) {
          setIsRunning(false)
        }
      }, iter * 500)
    }
  }

  useEffect(() => {
    generateData()
  }, [k])

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-2xl font-bold text-gray-900 mb-4">K-Means Clustering</h3>
        <p className="text-gray-600 mb-6">
          Watch how K-means algorithm groups similar data points into clusters.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <div className="bg-gray-50 rounded-lg p-4 h-80">
            <svg width="100%" height="300" viewBox="0 0 400 300">
              {/* Data points */}
              {points.map((point, i) => (
                <circle
                  key={i}
                  cx={point.x}
                  cy={point.y}
                  r="4"
                  fill={point.cluster >= 0 ? colors[point.cluster] : '#6b7280'}
                  opacity="0.7"
                />
              ))}
              
              {/* Centroids */}
              {centroids.map((centroid, i) => (
                <g key={i}>
                  <circle
                    cx={centroid.x}
                    cy={centroid.y}
                    r="8"
                    fill={colors[i]}
                    stroke="#000"
                    strokeWidth="2"
                  />
                  <text
                    x={centroid.x}
                    y={centroid.y + 3}
                    textAnchor="middle"
                    className="text-xs font-bold fill-white"
                  >
                    {i + 1}
                  </text>
                </g>
              ))}
            </svg>
          </div>
        </div>
        
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Number of Clusters (k): {k}
            </label>
            <input
              type="range"
              min="2"
              max="5"
              value={k}
              onChange={(e) => setK(parseInt(e.target.value))}
              className="w-full"
              disabled={isRunning}
            />
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <h4 className="font-semibold text-gray-900 mb-2">Legend:</h4>
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                <div className="w-4 h-4 bg-gray-400 rounded-full"></div>
                <span className="text-sm">Unassigned points</span>
              </div>
              {Array.from({ length: k }, (_, i) => (
                <div key={i} className="flex items-center space-x-2">
                  <div 
                    className="w-4 h-4 rounded-full border-2 border-black"
                    style={{ backgroundColor: colors[i] }}
                  ></div>
                  <span className="text-sm">Cluster {i + 1} & Centroid</span>
                </div>
              ))}
            </div>
          </div>
          
          <div className="flex space-x-3">
            <button
              onClick={runKMeans}
              disabled={isRunning}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {isRunning ? 'Running...' : 'Run K-Means'}
            </button>
            <button
              onClick={generateData}
              className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
            >
              New Data
            </button>
          </div>
          
          <div className="bg-green-50 rounded-lg p-4">
            <h5 className="font-medium text-green-900 mb-2">🎯 Algorithm Steps:</h5>
            <ol className="text-green-800 text-sm space-y-1 list-decimal list-inside">
              <li>Initialize k random centroids</li>
              <li>Assign each point to nearest centroid</li>
              <li>Move centroids to cluster centers</li>
              <li>Repeat until convergence</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  )
}

function DecisionBoundaryViz() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-2xl font-bold text-gray-900 mb-4">Decision Boundaries</h3>
        <p className="text-gray-600 mb-6">
          Visualize how different algorithms create decision boundaries to separate classes.
        </p>
      </div>
      
      <div className="bg-gray-50 rounded-lg p-8 text-center">
        <div className="text-6xl mb-4">🚧</div>
        <h4 className="text-xl font-semibold text-gray-900 mb-2">Coming Soon</h4>
        <p className="text-gray-600">
          Interactive decision boundary visualization with multiple algorithms
        </p>
      </div>
    </div>
  )
}

function BackpropViz() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-2xl font-bold text-gray-900 mb-4">Backpropagation Visualization</h3>
        <p className="text-gray-600 mb-6">
          See how gradients flow backward through a neural network during training.
        </p>
      </div>
      
      <div className="bg-gray-50 rounded-lg p-8 text-center">
        <div className="text-6xl mb-4">🔄</div>
        <h4 className="text-xl font-semibold text-gray-900 mb-2">Coming Soon</h4>
        <p className="text-gray-600">
          Step-by-step backpropagation with gradient flow visualization
        </p>
      </div>
    </div>
  )
}

function AttentionViz() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-2xl font-bold text-gray-900 mb-4">Attention Mechanism</h3>
        <p className="text-gray-600 mb-6">
          Explore how attention mechanisms focus on relevant parts of the input.
        </p>
      </div>
      
      <div className="bg-gray-50 rounded-lg p-8 text-center">
        <div className="text-6xl mb-4">👁️</div>
        <h4 className="text-xl font-semibold text-gray-900 mb-2">Coming Soon</h4>
        <p className="text-gray-600">
          Interactive attention weight visualization for transformers
        </p>
      </div>
    </div>
  )
} 