'use client'

import { useEffect, useState } from 'react'

interface ResearchPaper {
  id: string
  title: string
  authors: string[]
  published: string
  link: string
  category: string
  importance: 'high' | 'medium' | 'low'
}

export default function Footer() {
  const [featuredPapers, setFeaturedPapers] = useState<ResearchPaper[]>([])
  const [loading, setLoading] = useState(false)

  const fetchFeaturedPapers = async () => {
    setLoading(true)
    try {
      // Fetch highly cited and recent papers from arXiv
      const importantQueries = [
        'transformer',
        'attention mechanism',
        'large language model',
        'diffusion model',
        'reinforcement learning'
      ]
      
      const randomQuery = importantQueries[Math.floor(Math.random() * importantQueries.length)]
      const response = await fetch(
        `https://export.arxiv.org/api/query?search_query=all:${randomQuery}&start=0&max_results=5&sortBy=submittedDate&sortOrder=descending`
      )
      
      const xmlText = await response.text()
      const parser = new DOMParser()
      const xmlDoc = parser.parseFromString(xmlText, 'text/xml')
      const entries = xmlDoc.querySelectorAll('entry')
      
      const papers: ResearchPaper[] = Array.from(entries).map((entry, index) => {
        const title = entry.querySelector('title')?.textContent?.trim() || ''
        const published = entry.querySelector('published')?.textContent || ''
        const link = entry.querySelector('id')?.textContent || ''
        const authors = Array.from(entry.querySelectorAll('author name')).map(
          author => author.textContent || ''
        )
        
        // Determine importance based on keywords
        const titleLower = title.toLowerCase()
        let importance: 'high' | 'medium' | 'low' = 'medium'
        
        if (titleLower.includes('transformer') || titleLower.includes('gpt') || 
            titleLower.includes('attention') || titleLower.includes('diffusion')) {
          importance = 'high'
        } else if (titleLower.includes('neural') || titleLower.includes('learning')) {
          importance = 'medium'
        } else {
          importance = 'low'
        }
        
        return {
          id: `footer-paper-${index}`,
          title: title.replace(/\s+/g, ' ').substring(0, 80) + (title.length > 80 ? '...' : ''),
          authors: authors.slice(0, 2),
          published: new Date(published).toLocaleDateString(),
          link,
          category: randomQuery.replace(' ', '-'),
          importance
        }
      })
      
      setFeaturedPapers(papers)
    } catch (error) {
      console.error('Error fetching featured papers:', error)
      // Fallback to curated important papers
      setFeaturedPapers([
        {
          id: 'featured-1',
          title: 'Attention Is All You Need',
          authors: ['Vaswani et al.'],
          published: '2017-06-12',
          link: 'https://arxiv.org/abs/1706.03762',
          category: 'transformer',
          importance: 'high'
        },
        {
          id: 'featured-2',
          title: 'Language Models are Few-Shot Learners',
          authors: ['Brown et al.'],
          published: '2020-05-28',
          link: 'https://arxiv.org/abs/2005.14165',
          category: 'llm',
          importance: 'high'
        },
        {
          id: 'featured-3',
          title: 'Denoising Diffusion Probabilistic Models',
          authors: ['Ho et al.'],
          published: '2020-06-19',
          link: 'https://arxiv.org/abs/2006.11239',
          category: 'diffusion',
          importance: 'high'
        }
      ])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchFeaturedPapers()
  }, [])

  const getImportanceColor = (importance: string) => {
    switch (importance) {
      case 'high': return 'bg-red-100 text-red-700'
      case 'medium': return 'bg-yellow-100 text-yellow-700'
      default: return 'bg-gray-100 text-gray-700'
    }
  }

  return (
    <footer className="bg-gray-900 text-white py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center space-x-2 mb-4">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">ML</span>
              </div>
              <span className="font-bold text-xl">Academy</span>
            </div>
            <p className="text-gray-300 mb-4 max-w-md">
              Comprehensive machine learning and deep learning education platform. 
              From mathematical foundations to cutting-edge AI applications.
            </p>
            
            {/* Research Papers Section */}
            <div className="mt-6">
              <div className="flex items-center justify-between mb-3">
                <h4 className="font-semibold text-lg">🔬 Featured Research</h4>
                <button
                  onClick={fetchFeaturedPapers}
                  disabled={loading}
                  className="text-xs text-blue-400 hover:text-blue-300 transition-colors disabled:opacity-50"
                >
                  {loading ? '↻' : '🔄'}
                </button>
              </div>
              
              {loading ? (
                <div className="text-gray-400 text-sm">Loading latest papers...</div>
              ) : (
                <div className="space-y-3">
                  {featuredPapers.slice(0, 3).map((paper) => (
                    <div key={paper.id} className="border-l-2 border-blue-500 pl-3">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <a
                            href={paper.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-sm font-medium text-gray-200 hover:text-white transition-colors line-clamp-2"
                          >
                            {paper.title}
                          </a>
                          <div className="flex items-center space-x-2 mt-1">
                            <span className="text-xs text-gray-400">
                              {paper.authors.join(', ')}
                            </span>
                            <span className={`text-xs px-2 py-0.5 rounded-full ${getImportanceColor(paper.importance)}`}>
                              {paper.importance}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
              
              <div className="mt-4 text-xs text-gray-400">
                Papers sourced from arXiv • Updated daily
              </div>
            </div>
          </div>
          
          <div>
            <h3 className="font-semibold text-lg mb-4">Learning Paths</h3>
            <ul className="space-y-2 text-gray-300">
              <li><a href="#" className="hover:text-white transition-colors">Mathematics</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Machine Learning</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Deep Learning</a></li>
              <li><a href="#" className="hover:text-white transition-colors">AI History</a></li>
            </ul>
            
            <h4 className="font-semibold text-md mt-6 mb-3">🏆 Top Conferences</h4>
            <ul className="space-y-1 text-gray-300 text-sm">
              <li><a href="https://nips.cc" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">NeurIPS</a></li>
              <li><a href="https://icml.cc" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">ICML</a></li>
              <li><a href="https://iclr.cc" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">ICLR</a></li>
              <li><a href="https://aaai.org" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">AAAI</a></li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-semibold text-lg mb-4">Resources</h3>
            <ul className="space-y-2 text-gray-300">
              <li><a href="#" className="hover:text-white transition-colors">Interactive Demos</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Practice Problems</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Code Examples</a></li>
              <li><a href="https://arxiv.org/list/cs.LG/recent" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">Research Papers</a></li>
            </ul>
            
            <h4 className="font-semibold text-md mt-6 mb-3">📚 Key Datasets</h4>
            <ul className="space-y-1 text-gray-300 text-sm">
              <li><a href="https://www.image-net.org" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">ImageNet</a></li>
              <li><a href="https://www.kaggle.com/datasets" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">Kaggle Datasets</a></li>
              <li><a href="https://huggingface.co/datasets" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">HuggingFace</a></li>
              <li><a href="https://paperswithcode.com/datasets" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">Papers with Code</a></li>
            </ul>
          </div>
        </div>
        
        <div className="border-t border-gray-700 mt-8 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <p className="text-gray-400 text-center md:text-left">
              &copy; 2024 ML Academy. Built for educational purposes.
            </p>
            <div className="flex items-center space-x-4 mt-4 md:mt-0">
              <a href="https://arxiv.org" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white transition-colors text-sm">
                arXiv
              </a>
              <a href="https://paperswithcode.com" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white transition-colors text-sm">
                Papers with Code
              </a>
              <a href="https://scholar.google.com" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-white transition-colors text-sm">
                Google Scholar
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
} 