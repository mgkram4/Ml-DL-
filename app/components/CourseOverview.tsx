'use client'

export default function CourseOverview() {
  const courses = [
    {
      title: "Mathematical Foundations",
      description: "Linear algebra, calculus, probability, and statistics",
      topics: ["Vectors & Matrices", "Derivatives & Gradients", "Probability Theory", "Statistical Inference"],
      difficulty: "Beginner",
      duration: "4-6 weeks",
      color: "blue"
    },
    {
      title: "Machine Learning Fundamentals",
      description: "Core ML algorithms and concepts",
      topics: ["Supervised Learning", "Unsupervised Learning", "Model Evaluation", "Feature Engineering"],
      difficulty: "Intermediate",
      duration: "6-8 weeks",
      color: "green"
    },
    {
      title: "Deep Learning",
      description: "Neural networks and advanced architectures",
      topics: ["Neural Networks", "CNNs", "RNNs", "Transformers"],
      difficulty: "Advanced",
      duration: "8-10 weeks",
      color: "purple"
    },
    {
      title: "Specialized Topics",
      description: "Cutting-edge AI applications",
      topics: ["Computer Vision", "NLP", "Reinforcement Learning", "GANs"],
      difficulty: "Expert",
      duration: "6-8 weeks",
      color: "red"
    }
  ]

  const getColorClasses = (color: string) => {
    const colorMap = {
      blue: "bg-blue-50 border-blue-200 text-blue-800",
      green: "bg-green-50 border-green-200 text-green-800",
      purple: "bg-purple-50 border-purple-200 text-purple-800",
      red: "bg-red-50 border-red-200 text-red-800"
    }
    return colorMap[color as keyof typeof colorMap] || colorMap.blue
  }

  return (
    <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gray-50">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">Complete Learning Path</h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Our comprehensive curriculum takes you from mathematical foundations to advanced AI applications
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {courses.map((course, index) => (
            <div key={index} className="bg-white rounded-xl shadow-lg p-6 card-hover">
              <div className="flex items-center justify-between mb-4">
                <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getColorClasses(course.color)}`}>
                  {course.difficulty}
                </span>
                <span className="text-sm text-gray-500">{course.duration}</span>
              </div>
              
              <h3 className="text-xl font-semibold text-gray-900 mb-3">{course.title}</h3>
              <p className="text-gray-600 mb-4">{course.description}</p>
              
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-900">Key Topics:</h4>
                <ul className="space-y-1">
                  {course.topics.map((topic, topicIndex) => (
                    <li key={topicIndex} className="text-sm text-gray-600 flex items-center">
                      <span className="w-1.5 h-1.5 bg-blue-500 rounded-full mr-2"></span>
                      {topic}
                    </li>
                  ))}
                </ul>
              </div>
              
              <button className="w-full mt-6 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors duration-200">
                Start Module
              </button>
            </div>
          ))}
        </div>

        {/* Progress indicator */}
        <div className="mt-12 text-center">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Your Learning Journey</h3>
          <div className="max-w-2xl mx-auto">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-600">Progress</span>
              <span className="text-sm text-gray-600">0% Complete</span>
            </div>
            <div className="progress-bar">
              <div className="progress-fill" style={{ width: '0%' }}></div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
} 