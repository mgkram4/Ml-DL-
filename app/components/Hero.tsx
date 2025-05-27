'use client'

export default function Hero() {
  return (
    <section className="pt-20 pb-16 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-6">
            Master{' '}
            <span className="gradient-text">Machine Learning</span>
            {' '}&{' '}
            <span className="gradient-text">Deep Learning</span>
          </h1>
          
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            From fundamental mathematics to cutting-edge neural networks. 
            Learn everything you need to know about AI, with interactive examples, 
            comprehensive theory, and hands-on practice.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
            <button className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200 transform hover:scale-105">
              Start Learning
            </button>
            <button className="px-8 py-4 border-2 border-blue-600 text-blue-600 font-semibold rounded-lg hover:bg-blue-50 transition-all duration-200">
              Explore Curriculum
            </button>
          </div>

          {/* Feature highlights */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-16">
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">📊</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Interactive Visualizations</h3>
              <p className="text-gray-600">See algorithms in action with dynamic, interactive demonstrations</p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🧮</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Mathematical Foundations</h3>
              <p className="text-gray-600">Build strong mathematical intuition from linear algebra to calculus</p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🚀</span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Practical Applications</h3>
              <p className="text-gray-600">Apply your knowledge to real-world problems and projects</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
} 