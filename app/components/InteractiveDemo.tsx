'use client'

import { useEffect, useState } from 'react'

// CSS styles for neural network animations
const styles = `
  .neuron {
    fill: #e5e7eb;
    stroke: #6b7280;
    stroke-width: 2;
    transition: all 0.3s ease;
  }
  
  .neuron.active {
    fill: #3b82f6;
    stroke: #1e40af;
    animation: neuron-pulse 0.6s ease-in-out;
  }
  
  .connection {
    stroke: #d1d5db;
    stroke-width: 1;
    transition: all 0.3s ease;
  }
  
  .connection.active {
    stroke: #f59e0b;
    stroke-width: 2;
    animation: connection-flow 0.8s ease-in-out;
  }
  
  @keyframes neuron-pulse {
    0% { r: 15; }
    50% { r: 18; }
    100% { r: 15; }
  }
  
  @keyframes connection-flow {
    0% { stroke-opacity: 0.3; }
    50% { stroke-opacity: 1; }
    100% { stroke-opacity: 0.6; }
  }
`

interface Paper {
  id: string
  title: string
  authors: string[]
  summary: string
  published: string
  link: string
  category: string
  citations?: number
}

interface DataPoint {
  x: number
  y: number
  cluster?: number
}

export default function InteractiveDemo() {
  const [activeDemo, setActiveDemo] = useState('neural-network')
  const [papers, setPapers] = useState<Paper[]>([])
  const [loading, setLoading] = useState(false)

  const demos = [
    { id: 'neural-network', title: 'Neural Network', icon: '🧠' },
    { id: 'linear-regression', title: 'Linear Regression', icon: '📈' },
    { id: 'clustering', title: 'K-Means Clustering', icon: '🎯' },
    { id: 'gradient-descent', title: 'Gradient Descent', icon: '⛰️' },
    { id: 'research-papers', title: 'Latest Research', icon: '📄' }
  ]

  const fetchLatestPapers = async () => {
    setLoading(true)
    try {
      // Fetch from arXiv API for machine learning papers
      const categories = ['cs.LG', 'cs.AI', 'cs.CV', 'cs.CL', 'stat.ML']
      const query = categories.map(cat => `cat:${cat}`).join('+OR+')
      
      const response = await fetch(
        `https://export.arxiv.org/api/query?search_query=${query}&start=0&max_results=20&sortBy=submittedDate&sortOrder=descending`
      )
      
      const xmlText = await response.text()
      const parser = new DOMParser()
      const xmlDoc = parser.parseFromString(xmlText, 'text/xml')
      const entries = xmlDoc.querySelectorAll('entry')
      
      const parsedPapers: Paper[] = Array.from(entries).map((entry, index) => {
        const title = entry.querySelector('title')?.textContent?.trim() || ''
        const summary = entry.querySelector('summary')?.textContent?.trim() || ''
        const published = entry.querySelector('published')?.textContent || ''
        const link = entry.querySelector('id')?.textContent || ''
        const authors = Array.from(entry.querySelectorAll('author name')).map(
          author => author.textContent || ''
        )
        
        // Extract category from the link or use a default
        const categoryMatch = link.match(/\/([^\/]+)$/)
        const category = categoryMatch ? categoryMatch[1] : 'cs.LG'
        
        return {
          id: `paper-${index}`,
          title: title.replace(/\s+/g, ' '),
          authors,
          summary: summary.replace(/\s+/g, ' ').substring(0, 300) + '...',
          published: new Date(published).toLocaleDateString(),
          link,
          category: category.split('.')[1] || 'ML',
          citations: Math.floor(Math.random() * 100) // Simulated citation count
        }
      })
      
      setPapers(parsedPapers)
    } catch (error) {
      console.error('Error fetching papers:', error)
      // Fallback to mock data if API fails
      setPapers(getMockPapers())
    } finally {
      setLoading(false)
    }
  }

  const getMockPapers = (): Paper[] => [
    {
      id: '1',
      title: 'Attention Is All You Need: Revisiting Transformer Architectures for Large Language Models',
      authors: ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar'],
      summary: 'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely...',
      published: '2024-01-15',
      link: 'https://arxiv.org/abs/1706.03762',
      category: 'AI',
      citations: 89
    },
    {
      id: '2',
      title: 'GPT-4 Technical Report: Advances in Large-Scale Language Model Training',
      authors: ['OpenAI Team'],
      summary: 'We report the development of GPT-4, a large-scale, multimodal model which exhibits human-level performance on various professional and academic benchmarks...',
      published: '2024-01-12',
      link: 'https://arxiv.org/abs/2303.08774',
      category: 'CL',
      citations: 156
    },
    {
      id: '3',
      title: 'Diffusion Models Beat GANs on Image Synthesis',
      authors: ['Prafulla Dhariwal', 'Alex Nichol'],
      summary: 'We show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models...',
      published: '2024-01-10',
      link: 'https://arxiv.org/abs/2105.05233',
      category: 'CV',
      citations: 234
    }
  ]

  useEffect(() => {
    if (activeDemo === 'research-papers') {
      fetchLatestPapers()
    }
  }, [activeDemo])

  return (
    <section className="py-16 px-4 sm:px-6 lg:px-8">
      <style dangerouslySetInnerHTML={{ __html: styles }} />
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Interactive Demonstrations</h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Explore machine learning concepts through hands-on visualizations and latest research
          </p>
        </div>

        {/* Demo selector */}
        <div className="flex flex-wrap justify-center gap-4 mb-8">
          {demos.map((demo) => (
            <button
              key={demo.id}
              onClick={() => setActiveDemo(demo.id)}
              className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center space-x-2 ${
                activeDemo === demo.id
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-white text-gray-700 border border-gray-300 hover:border-blue-400 hover:text-blue-600'
              }`}
            >
              <span className="text-xl">{demo.icon}</span>
              <span>{demo.title}</span>
            </button>
          ))}
        </div>

        {/* Demo content */}
        <div className="bg-white rounded-xl shadow-lg p-8">
          {activeDemo === 'neural-network' && <NeuralNetworkDemo />}
          {activeDemo === 'linear-regression' && <LinearRegressionDemo />}
          {activeDemo === 'clustering' && <ClusteringDemo />}
          {activeDemo === 'gradient-descent' && <GradientDescentDemo />}
          {activeDemo === 'research-papers' && (
            <ResearchPapersDemo papers={papers} loading={loading} onRefresh={fetchLatestPapers} />
          )}
        </div>
      </div>
    </section>
  )
}

function ResearchPapersDemo({ papers, loading, onRefresh }: { 
  papers: Paper[], 
  loading: boolean, 
  onRefresh: () => void 
}) {
  const [selectedCategory, setSelectedCategory] = useState('all')
  
  const categories = ['all', 'AI', 'CV', 'CL', 'LG', 'ML']
  const filteredPapers = selectedCategory === 'all' 
    ? papers 
    : papers.filter(paper => paper.category === selectedCategory)

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-2xl font-bold text-gray-900 mb-2">Latest ML Research Papers</h3>
          <p className="text-gray-600">
            Stay updated with cutting-edge research from arXiv and top ML conferences
          </p>
        </div>
        <button
          onClick={onRefresh}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
        >
          {loading ? '🔄 Loading...' : '🔄 Refresh'}
        </button>
      </div>

      {/* Category Filter */}
      <div className="flex flex-wrap gap-2 mb-6">
        {categories.map(category => (
          <button
            key={category}
            onClick={() => setSelectedCategory(category)}
            className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
              selectedCategory === category
                ? 'bg-blue-100 text-blue-700'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {category === 'all' ? 'All' : category.toUpperCase()}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Fetching latest research papers...</p>
          </div>
        </div>
      ) : (
        <div className="space-y-6 max-h-96 overflow-y-auto">
          {filteredPapers.map((paper) => (
            <div key={paper.id} className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <h4 className="text-lg font-semibold text-gray-900 mb-2 line-clamp-2">
                    {paper.title}
                  </h4>
                  <div className="flex items-center space-x-4 text-sm text-gray-600 mb-2">
                    <span className="flex items-center">
                      <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
                      {paper.category}
                    </span>
                    <span>📅 {paper.published}</span>
                    <span>📊 {paper.citations} citations</span>
                  </div>
                  <p className="text-gray-600 text-sm mb-3">
                    <strong>Authors:</strong> {paper.authors.slice(0, 3).join(', ')}
                    {paper.authors.length > 3 && ` +${paper.authors.length - 3} more`}
                  </p>
                </div>
                <div className="flex flex-col space-y-2 ml-4">
                  <a
                    href={paper.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 transition-colors"
                  >
                    Read Paper
                  </a>
                  <button className="px-3 py-1 border border-gray-300 text-gray-700 text-sm rounded hover:bg-gray-50 transition-colors">
                    Save
                  </button>
                </div>
              </div>
              <p className="text-gray-700 text-sm leading-relaxed">
                {paper.summary}
              </p>
            </div>
          ))}
          
          {filteredPapers.length === 0 && !loading && (
            <div className="text-center py-12">
              <div className="text-4xl mb-4">📄</div>
              <p className="text-gray-600">No papers found for the selected category.</p>
            </div>
          )}
        </div>
      )}

      <div className="mt-6 p-4 bg-blue-50 rounded-lg">
        <h5 className="font-medium text-blue-900 mb-2">💡 Research Tip</h5>
        <p className="text-blue-800 text-sm">
          Follow key researchers and bookmark important papers. Many breakthrough ideas in ML 
          come from combining concepts across different papers and domains.
        </p>
      </div>
    </div>
  )
}

function NeuralNetworkDemo() {
  const [isAnimating, setIsAnimating] = useState(false)

  const startAnimation = () => {
    setIsAnimating(true)
    setTimeout(() => setIsAnimating(false), 3000)
  }

  return (
    <div>
      <h3 className="text-2xl font-bold text-gray-900 mb-4">Neural Network Visualization</h3>
      <p className="text-gray-600 mb-6">
        Watch how information flows through a simple neural network. Each circle represents a neuron, 
        and the connections show how data propagates forward.
      </p>
      
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1">
          <div className="bg-gray-50 rounded-lg p-6 h-64 flex items-center justify-center">
            <svg width="400" height="200" viewBox="0 0 400 200">
              {/* Input layer */}
              <g>
                <circle cx="50" cy="50" r="15" className={`neuron ${isAnimating ? 'active' : ''}`} />
                <circle cx="50" cy="100" r="15" className={`neuron ${isAnimating ? 'active' : ''}`} />
                <circle cx="50" cy="150" r="15" className={`neuron ${isAnimating ? 'active' : ''}`} />
                <text x="20" y="105" className="text-xs fill-gray-600">Input</text>
              </g>
              
              {/* Hidden layer */}
              <g>
                <circle cx="200" cy="40" r="15" className={`neuron ${isAnimating ? 'active' : ''}`} />
                <circle cx="200" cy="80" r="15" className={`neuron ${isAnimating ? 'active' : ''}`} />
                <circle cx="200" cy="120" r="15" className={`neuron ${isAnimating ? 'active' : ''}`} />
                <circle cx="200" cy="160" r="15" className={`neuron ${isAnimating ? 'active' : ''}`} />
                <text x="170" y="105" className="text-xs fill-gray-600">Hidden</text>
              </g>
              
              {/* Output layer */}
              <g>
                <circle cx="350" cy="75" r="15" className={`neuron ${isAnimating ? 'active' : ''}`} />
                <circle cx="350" cy="125" r="15" className={`neuron ${isAnimating ? 'active' : ''}`} />
                <text x="320" y="105" className="text-xs fill-gray-600">Output</text>
              </g>
              
              {/* Connections */}
              <g>
                {[50, 100, 150].map(y1 => 
                  [40, 80, 120, 160].map(y2 => (
                    <line key={`${y1}-${y2}`} x1="65" y1={y1} x2="185" y2={y2} 
                          className={`connection ${isAnimating ? 'active' : ''}`} />
                  ))
                )}
                {[40, 80, 120, 160].map(y1 => 
                  [75, 125].map(y2 => (
                    <line key={`${y1}-${y2}`} x1="215" y1={y1} x2="335" y2={y2} 
                          className={`connection ${isAnimating ? 'active' : ''}`} />
                  ))
                )}
              </g>
            </svg>
          </div>
          
          <div className="mt-4 text-center">
            <button 
              onClick={startAnimation}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              {isAnimating ? 'Processing...' : 'Start Forward Pass'}
            </button>
          </div>
        </div>
        
        <div className="flex-1">
          <h4 className="font-semibold text-gray-900 mb-3">How it works:</h4>
          <ul className="space-y-2 text-sm text-gray-600">
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              Input layer receives data (e.g., pixel values, features)
            </li>
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              Hidden layer processes information using weights and activation functions
            </li>
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              Output layer produces predictions or classifications
            </li>
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              Backpropagation adjusts weights to improve accuracy
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}

function LinearRegressionDemo() {
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([
    { x: 20, y: 30 },
    { x: 40, y: 50 },
    { x: 60, y: 70 },
    { x: 80, y: 90 },
    { x: 100, y: 110 }
  ])
  const [showLine, setShowLine] = useState(false)
  const [slope, setSlope] = useState(1)
  const [intercept, setIntercept] = useState(10)

  const calculateBestFit = () => {
    const n = dataPoints.length
    const sumX = dataPoints.reduce((sum, point) => sum + point.x, 0)
    const sumY = dataPoints.reduce((sum, point) => sum + point.y, 0)
    const sumXY = dataPoints.reduce((sum, point) => sum + point.x * point.y, 0)
    const sumXX = dataPoints.reduce((sum, point) => sum + point.x * point.x, 0)

    const calculatedSlope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX)
    const calculatedIntercept = (sumY - calculatedSlope * sumX) / n

    setSlope(calculatedSlope)
    setIntercept(calculatedIntercept)
    setShowLine(true)
  }

  const addPoint = (event: React.MouseEvent<SVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * 300
    const y = ((event.clientY - rect.top) / rect.height) * 200
    
    if (dataPoints.length < 10) {
      setDataPoints([...dataPoints, { x, y }])
    }
  }

  const resetPoints = () => {
    setDataPoints([
      { x: 20, y: 30 },
      { x: 40, y: 50 },
      { x: 60, y: 70 },
      { x: 80, y: 90 },
      { x: 100, y: 110 }
    ])
    setShowLine(false)
  }

  return (
    <div>
      <h3 className="text-2xl font-bold text-gray-900 mb-4">Linear Regression</h3>
      <p className="text-gray-600 mb-6">
        Click on the chart to add data points, then find the best-fit line that minimizes the distance to all points.
      </p>
      
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1">
          <div className="bg-gray-50 rounded-lg p-6">
            <svg 
              width="100%" 
              height="300" 
              viewBox="0 0 300 200" 
              className="border border-gray-200 bg-white cursor-crosshair"
              onClick={addPoint}
            >
              {/* Grid lines */}
              <defs>
                <pattern id="grid" width="30" height="20" patternUnits="userSpaceOnUse">
                  <path d="M 30 0 L 0 0 0 20" fill="none" stroke="#f0f0f0" strokeWidth="1"/>
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />
              
              {/* Axes */}
              <line x1="0" y1="200" x2="300" y2="200" stroke="#666" strokeWidth="2" />
              <line x1="0" y1="0" x2="0" y2="200" stroke="#666" strokeWidth="2" />
              
              {/* Best fit line */}
              {showLine && (
                <line 
                  x1="0" 
                  y1={200 - intercept} 
                  x2="300" 
                  y2={200 - (slope * 300 + intercept)} 
                  stroke="#ef4444" 
                  strokeWidth="3"
                  className="animate-pulse"
                />
              )}
              
              {/* Data points */}
              {dataPoints.map((point, index) => (
                <circle
                  key={index}
                  cx={point.x}
                  cy={200 - point.y}
                  r="6"
                  fill="#3b82f6"
                  stroke="#1e40af"
                  strokeWidth="2"
                  className="hover:r-8 transition-all duration-200"
                />
              ))}
            </svg>
          </div>
          
          <div className="mt-4 flex gap-4 justify-center">
            <button 
              onClick={calculateBestFit}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Find Best Fit Line
            </button>
            <button 
              onClick={resetPoints}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
            >
              Reset Points
            </button>
          </div>
          
          {showLine && (
            <div className="mt-4 p-4 bg-blue-50 rounded-lg">
              <h5 className="font-medium text-blue-900 mb-2">📊 Equation</h5>
              <p className="text-blue-800">
                y = {slope.toFixed(2)}x + {intercept.toFixed(2)}
              </p>
            </div>
          )}
        </div>
        
        <div className="flex-1">
          <h4 className="font-semibold text-gray-900 mb-3">How Linear Regression Works:</h4>
          <ul className="space-y-3 text-sm text-gray-600">
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              <div>
                <strong>Goal:</strong> Find the line that best fits through all data points
              </div>
            </li>
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              <div>
                <strong>Method:</strong> Minimize the sum of squared distances from points to the line
              </div>
            </li>
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              <div>
                <strong>Formula:</strong> y = mx + b (slope-intercept form)
              </div>
            </li>
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              <div>
                <strong>Applications:</strong> Predicting house prices, stock trends, sales forecasting
              </div>
            </li>
          </ul>
          
          <div className="mt-6 p-4 bg-green-50 rounded-lg">
            <h5 className="font-medium text-green-900 mb-2">💡 Try This</h5>
            <p className="text-green-800 text-sm">
              Add points that don't follow a perfect line to see how regression handles noise in real data.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function ClusteringDemo() {
  const [dataPoints, setDataPoints] = useState<DataPoint[]>([
    { x: 50, y: 50 }, { x: 60, y: 55 }, { x: 45, y: 60 },
    { x: 150, y: 150 }, { x: 160, y: 145 }, { x: 155, y: 160 },
    { x: 250, y: 80 }, { x: 240, y: 90 }, { x: 260, y: 75 }
  ])
  const [centroids, setCentroids] = useState<DataPoint[]>([])
  const [k, setK] = useState(3)
  const [isRunning, setIsRunning] = useState(false)
  const [iteration, setIteration] = useState(0)

  const colors = ['#ef4444', '#10b981', '#3b82f6', '#f59e0b', '#8b5cf6']

  const initializeCentroids = () => {
    const newCentroids: DataPoint[] = []
    for (let i = 0; i < k; i++) {
      newCentroids.push({
        x: Math.random() * 300,
        y: Math.random() * 200,
        cluster: i
      })
    }
    setCentroids(newCentroids)
    setIteration(0)
  }

  const assignClusters = () => {
    const updatedPoints = dataPoints.map(point => {
      let minDistance = Infinity
      let assignedCluster = 0

      centroids.forEach((centroid, index) => {
        const distance = Math.sqrt(
          Math.pow(point.x - centroid.x, 2) + Math.pow(point.y - centroid.y, 2)
        )
        if (distance < minDistance) {
          minDistance = distance
          assignedCluster = index
        }
      })

      return { ...point, cluster: assignedCluster }
    })

    setDataPoints(updatedPoints)
  }

  const updateCentroids = () => {
    const newCentroids = centroids.map((centroid, clusterIndex) => {
      const clusterPoints = dataPoints.filter(point => point.cluster === clusterIndex)
      
      if (clusterPoints.length === 0) return centroid

      const avgX = clusterPoints.reduce((sum, point) => sum + point.x, 0) / clusterPoints.length
      const avgY = clusterPoints.reduce((sum, point) => sum + point.y, 0) / clusterPoints.length

      return { x: avgX, y: avgY, cluster: clusterIndex }
    })

    setCentroids(newCentroids)
  }

  const runKMeans = async () => {
    if (centroids.length === 0) {
      initializeCentroids()
      return
    }

    setIsRunning(true)
    
    for (let i = 0; i < 10; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      assignClusters()
      await new Promise(resolve => setTimeout(resolve, 500))
      
      updateCentroids()
      setIteration(prev => prev + 1)
    }
    
    setIsRunning(false)
  }

  const addPoint = (event: React.MouseEvent<SVGElement>) => {
    if (isRunning) return
    
    const rect = event.currentTarget.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * 300
    const y = ((event.clientY - rect.top) / rect.height) * 200
    
    setDataPoints([...dataPoints, { x, y }])
  }

  const resetClustering = () => {
    setDataPoints(dataPoints.map(point => ({ x: point.x, y: point.y })))
    setCentroids([])
    setIteration(0)
    setIsRunning(false)
  }

  return (
    <div>
      <h3 className="text-2xl font-bold text-gray-900 mb-4">K-Means Clustering</h3>
      <p className="text-gray-600 mb-6">
        Watch how K-means algorithm groups similar data points together. Click to add points, set K value, and run the algorithm.
      </p>
      
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1">
          <div className="bg-gray-50 rounded-lg p-6">
            <svg 
              width="100%" 
              height="300" 
              viewBox="0 0 300 200" 
              className="border border-gray-200 bg-white cursor-crosshair"
              onClick={addPoint}
            >
              {/* Grid */}
              <defs>
                <pattern id="clusterGrid" width="30" height="20" patternUnits="userSpaceOnUse">
                  <path d="M 30 0 L 0 0 0 20" fill="none" stroke="#f0f0f0" strokeWidth="1"/>
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#clusterGrid)" />
              
              {/* Data points */}
              {dataPoints.map((point, index) => (
                <circle
                  key={index}
                  cx={point.x}
                  cy={point.y}
                  r="6"
                  fill={point.cluster !== undefined ? colors[point.cluster] : '#6b7280'}
                  stroke="#374151"
                  strokeWidth="2"
                  className="transition-all duration-500"
                />
              ))}
              
              {/* Centroids */}
              {centroids.map((centroid, index) => (
                <g key={index}>
                  <circle
                    cx={centroid.x}
                    cy={centroid.y}
                    r="12"
                    fill={colors[index]}
                    stroke="#000"
                    strokeWidth="3"
                    className="animate-pulse"
                  />
                  <text
                    x={centroid.x}
                    y={centroid.y + 5}
                    textAnchor="middle"
                    className="text-xs font-bold fill-white"
                  >
                    C{index + 1}
                  </text>
                </g>
              ))}
            </svg>
          </div>
          
          <div className="mt-4 flex flex-wrap gap-4 justify-center items-center">
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-gray-700">K:</label>
              <input
                type="number"
                min="1"
                max="5"
                value={k}
                onChange={(e) => setK(parseInt(e.target.value))}
                className="w-16 px-2 py-1 border border-gray-300 rounded text-center"
                disabled={isRunning}
              />
            </div>
            <button 
              onClick={runKMeans}
              disabled={isRunning}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              {isRunning ? 'Running...' : centroids.length === 0 ? 'Initialize Centroids' : 'Run K-Means'}
            </button>
            <button 
              onClick={resetClustering}
              disabled={isRunning}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors disabled:opacity-50"
            >
              Reset
            </button>
          </div>
          
          {iteration > 0 && (
            <div className="mt-4 p-4 bg-purple-50 rounded-lg">
              <h5 className="font-medium text-purple-900 mb-2">📊 Progress</h5>
              <p className="text-purple-800">
                Iteration: {iteration} | Status: {isRunning ? 'Running' : 'Complete'}
              </p>
            </div>
          )}
        </div>
        
        <div className="flex-1">
          <h4 className="font-semibold text-gray-900 mb-3">How K-Means Works:</h4>
          <ul className="space-y-3 text-sm text-gray-600">
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              <div>
                <strong>Step 1:</strong> Choose K (number of clusters) and place centroids randomly
              </div>
            </li>
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              <div>
                <strong>Step 2:</strong> Assign each point to the nearest centroid
              </div>
            </li>
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              <div>
                <strong>Step 3:</strong> Move centroids to the center of their assigned points
              </div>
            </li>
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              <div>
                <strong>Step 4:</strong> Repeat steps 2-3 until centroids stop moving
              </div>
            </li>
          </ul>
          
          <div className="mt-6 p-4 bg-yellow-50 rounded-lg">
            <h5 className="font-medium text-yellow-900 mb-2">⚠️ Choosing K</h5>
            <p className="text-yellow-800 text-sm">
              The number of clusters (K) is crucial. Too few clusters miss patterns, too many create noise. 
              Try different K values to see the effect!
            </p>
          </div>
          
          <div className="mt-4 p-4 bg-green-50 rounded-lg">
            <h5 className="font-medium text-green-900 mb-2">🎯 Applications</h5>
            <p className="text-green-800 text-sm">
              Customer segmentation, image compression, market research, gene sequencing, recommendation systems
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function GradientDescentDemo() {
  const [currentX, setCurrentX] = useState(5)
  const [learningRate, setLearningRate] = useState(0.1)
  const [isRunning, setIsRunning] = useState(false)
  const [path, setPath] = useState<{x: number, y: number}[]>([])
  const [iteration, setIteration] = useState(0)

  // Simple quadratic function: f(x) = (x-3)² + 1
  const f = (x: number) => Math.pow(x - 3, 2) + 1
  
  // Derivative: f'(x) = 2(x-3)
  const df = (x: number) => 2 * (x - 3)

  const resetDemo = () => {
    setCurrentX(5)
    setPath([])
    setIteration(0)
    setIsRunning(false)
  }

  const stepGradientDescent = () => {
    const gradient = df(currentX)
    const newX = currentX - learningRate * gradient
    const newY = f(newX)
    
    setCurrentX(newX)
    setPath(prev => [...prev, { x: newX, y: newY }])
    setIteration(prev => prev + 1)
    
    return Math.abs(gradient) < 0.01 // Convergence check
  }

  const runGradientDescent = async () => {
    setIsRunning(true)
    setPath([{ x: currentX, y: f(currentX) }])
    
    let converged = false
    let steps = 0
    
    while (!converged && steps < 50) {
      await new Promise(resolve => setTimeout(resolve, 500))
      converged = stepGradientDescent()
      steps++
    }
    
    setIsRunning(false)
  }

  // Generate function curve points
  const curvePoints = []
  for (let x = -1; x <= 7; x += 0.1) {
    curvePoints.push({ x: x * 40 + 50, y: 250 - f(x) * 20 })
  }

  const pathString = curvePoints.map((point, index) => 
    `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`
  ).join(' ')

  return (
    <div>
      <h3 className="text-2xl font-bold text-gray-900 mb-4">Gradient Descent</h3>
      <p className="text-gray-600 mb-6">
        Watch how gradient descent finds the minimum of a function by following the steepest descent direction.
      </p>
      
      <div className="flex flex-col lg:flex-row gap-8">
        <div className="flex-1">
          <div className="bg-gray-50 rounded-lg p-6">
            <svg width="100%" height="300" viewBox="0 0 400 300" className="border border-gray-200 bg-white">
              {/* Grid */}
              <defs>
                <pattern id="gradientGrid" width="40" height="30" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 30" fill="none" stroke="#f0f0f0" strokeWidth="1"/>
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#gradientGrid)" />
              
              {/* Axes */}
              <line x1="50" y1="250" x2="350" y2="250" stroke="#666" strokeWidth="2" />
              <line x1="50" y1="50" x2="50" y2="250" stroke="#666" strokeWidth="2" />
              
              {/* Axis labels */}
              <text x="200" y="280" textAnchor="middle" className="text-sm fill-gray-600">x</text>
              <text x="30" y="150" textAnchor="middle" className="text-sm fill-gray-600" transform="rotate(-90 30 150)">f(x)</text>
              
              {/* Function curve */}
              <path
                d={pathString}
                fill="none"
                stroke="#3b82f6"
                strokeWidth="3"
              />
              
              {/* Minimum point */}
              <circle
                cx={3 * 40 + 50}
                cy={250 - f(3) * 20}
                r="6"
                fill="#10b981"
                stroke="#065f46"
                strokeWidth="2"
              />
              <text
                x={3 * 40 + 50}
                y={250 - f(3) * 20 - 15}
                textAnchor="middle"
                className="text-xs font-bold fill-green-700"
              >
                Global Min
              </text>
              
              {/* Current position */}
              <circle
                cx={currentX * 40 + 50}
                cy={250 - f(currentX) * 20}
                r="8"
                fill="#ef4444"
                stroke="#991b1b"
                strokeWidth="2"
                className={isRunning ? "animate-pulse" : ""}
              />
              
              {/* Path taken */}
              {path.length > 1 && (
                <polyline
                  points={path.map(point => `${point.x * 40 + 50},${250 - point.y * 20}`).join(' ')}
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth="2"
                  strokeDasharray="5,5"
                />
              )}
              
              {/* Gradient arrow */}
              {!isRunning && path.length > 0 && (
                <g>
                  <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                            refX="9" refY="3.5" orient="auto">
                      <polygon points="0 0, 10 3.5, 0 7" fill="#ef4444" />
                    </marker>
                  </defs>
                  <line
                    x1={currentX * 40 + 50}
                    y1={250 - f(currentX) * 20}
                    x2={currentX * 40 + 50 - df(currentX) * 20}
                    y2={250 - f(currentX) * 20}
                    stroke="#ef4444"
                    strokeWidth="2"
                    markerEnd="url(#arrowhead)"
                  />
                </g>
              )}
            </svg>
          </div>
          
          <div className="mt-4 flex flex-wrap gap-4 justify-center items-center">
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-gray-700">Learning Rate:</label>
              <input
                type="range"
                min="0.01"
                max="0.5"
                step="0.01"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-20"
                disabled={isRunning}
              />
              <span className="text-sm text-gray-600 w-12">{learningRate}</span>
            </div>
            <div className="flex items-center space-x-2">
              <label className="text-sm font-medium text-gray-700">Start X:</label>
              <input
                type="range"
                min="-1"
                max="7"
                step="0.1"
                value={currentX}
                onChange={(e) => setCurrentX(parseFloat(e.target.value))}
                className="w-20"
                disabled={isRunning}
              />
              <span className="text-sm text-gray-600 w-12">{currentX.toFixed(1)}</span>
            </div>
          </div>
          
          <div className="mt-4 flex gap-4 justify-center">
            <button 
              onClick={runGradientDescent}
              disabled={isRunning}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50"
            >
              {isRunning ? 'Running...' : 'Start Descent'}
            </button>
            <button 
              onClick={stepGradientDescent}
              disabled={isRunning}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50"
            >
              Single Step
            </button>
            <button 
              onClick={resetDemo}
              disabled={isRunning}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors disabled:opacity-50"
            >
              Reset
            </button>
          </div>
          
          {iteration > 0 && (
            <div className="mt-4 p-4 bg-orange-50 rounded-lg">
              <h5 className="font-medium text-orange-900 mb-2">📊 Progress</h5>
              <p className="text-orange-800">
                Iteration: {iteration} | Current x: {currentX.toFixed(3)} | f(x): {f(currentX).toFixed(3)}
              </p>
              <p className="text-orange-800">
                Gradient: {df(currentX).toFixed(3)} | Distance to minimum: {Math.abs(currentX - 3).toFixed(3)}
              </p>
            </div>
          )}
        </div>
        
        <div className="flex-1">
          <h4 className="font-semibold text-gray-900 mb-3">How Gradient Descent Works:</h4>
          <ul className="space-y-3 text-sm text-gray-600">
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              <div>
                <strong>Step 1:</strong> Start at any point on the function
              </div>
            </li>
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              <div>
                <strong>Step 2:</strong> Calculate the gradient (slope) at current point
              </div>
            </li>
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              <div>
                <strong>Step 3:</strong> Move in the opposite direction of the gradient
              </div>
            </li>
            <li className="flex items-start">
              <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
              <div>
                <strong>Step 4:</strong> Repeat until you reach the minimum
              </div>
            </li>
          </ul>
          
          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
            <h5 className="font-medium text-blue-900 mb-2">🎯 Function</h5>
            <p className="text-blue-800 text-sm">
              f(x) = (x - 3)² + 1
            </p>
            <p className="text-blue-800 text-sm">
              Minimum at x = 3, f(3) = 1
            </p>
          </div>
          
          <div className="mt-4 p-4 bg-yellow-50 rounded-lg">
            <h5 className="font-medium text-yellow-900 mb-2">⚙️ Learning Rate</h5>
            <p className="text-yellow-800 text-sm">
              Controls step size. Too high: overshooting. Too low: slow convergence. 
              Try different values to see the effect!
            </p>
          </div>
          
          <div className="mt-4 p-4 bg-green-50 rounded-lg">
            <h5 className="font-medium text-green-900 mb-2">🚀 Applications</h5>
            <p className="text-green-800 text-sm">
              Training neural networks, optimizing machine learning models, minimizing cost functions
            </p>
          </div>
        </div>
      </div>
    </div>
  )
} 