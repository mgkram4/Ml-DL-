'use client'

import { useState } from 'react'
import CourseOverview from './components/CourseOverview'
import DeepLearningContent from './components/DeepLearningContent'
import Footer from './components/Footer'
import Hero from './components/Hero'
import HistoryTimeline from './components/HistoryTimeline'
import InteractiveDemo from './components/InteractiveDemo'
import MachineLearningContent from './components/MachineLearningContent'
import MathPrerequisites from './components/MathPrerequisites'
import Navigation from './components/Navigation'
import QuizSystem from './components/QuizSystem'

export default function Home() {
  const [activeSection, setActiveSection] = useState('home')
  const [expandedFaq, setExpandedFaq] = useState<string | null>(null)
  const [selectedLearningPath, setSelectedLearningPath] = useState<string | null>(null)

  const features = [
    {
      icon: "🧠",
      title: "Interactive Learning",
      description: "Hands-on exercises and visualizations that make complex concepts easy to understand",
      details: "Our interactive approach includes live code examples, visual demonstrations, and step-by-step tutorials that adapt to your learning pace."
    },
    {
      icon: "📊",
      title: "Real-World Projects",
      description: "Build actual ML models and solve practical problems from day one",
      details: "Work on projects like image classification, natural language processing, and predictive analytics using real datasets from industry."
    },
    {
      icon: "🎯",
      title: "Personalized Path",
      description: "Adaptive curriculum that adjusts to your background and goals",
      details: "Whether you're a beginner or have some experience, our system creates a customized learning path that maximizes your progress."
    },
    {
      icon: "👥",
      title: "Expert Community",
      description: "Learn from industry professionals and connect with fellow learners",
      details: "Join a vibrant community of ML practitioners, researchers, and enthusiasts who share knowledge and collaborate on projects."
    },
    {
      icon: "🚀",
      title: "Career Ready",
      description: "Skills and portfolio projects that employers are looking for",
      details: "Graduate with a portfolio of projects and skills that directly translate to high-demand roles in AI and machine learning."
    },
    {
      icon: "🔬",
      title: "Cutting-Edge Content",
      description: "Stay current with the latest developments in AI and ML",
      details: "Our curriculum is continuously updated with the latest research, tools, and techniques from leading AI labs and companies."
    }
  ]

  const learningPaths = [
    {
      id: "beginner",
      title: "Beginner Path",
      duration: "3-6 months",
      description: "Perfect for those new to programming and mathematics",
      color: "bg-green-100 border-green-200 text-green-800",
      startSection: "mathematics",
      modules: [
        "Mathematical Foundations",
        "Python Programming",
        "Statistics & Probability",
        "Introduction to ML",
        "First ML Project"
      ],
      outcome: "Build your first machine learning model and understand core concepts"
    },
    {
      id: "intermediate",
      title: "Intermediate Path",
      duration: "4-8 months",
      description: "For those with programming experience wanting to specialize in ML",
      color: "bg-blue-100 border-blue-200 text-blue-800",
      startSection: "machine-learning",
      modules: [
        "Advanced Mathematics",
        "Machine Learning Algorithms",
        "Deep Learning Fundamentals",
        "Computer Vision",
        "Natural Language Processing"
      ],
      outcome: "Develop expertise in multiple ML domains and build a professional portfolio"
    },
    {
      id: "advanced",
      title: "Advanced Path",
      duration: "6-12 months",
      description: "For experienced developers ready for cutting-edge AI research",
      color: "bg-purple-100 border-purple-200 text-purple-800",
      startSection: "deep-learning",
      modules: [
        "Advanced Deep Learning",
        "Reinforcement Learning",
        "Generative AI",
        "MLOps & Production",
        "Research Project"
      ],
      outcome: "Master state-of-the-art techniques and contribute to AI research"
    }
  ]

  const testimonials = [
    {
      name: "Sarah Chen",
      role: "ML Engineer at Google",
      image: "👩‍💻",
      quote: "This course transformed my career. The hands-on approach and real-world projects gave me the confidence to transition into machine learning.",
      rating: 5
    },
    {
      name: "Marcus Rodriguez",
      role: "Data Scientist at Netflix",
      image: "👨‍🔬",
      quote: "The mathematical foundations section was incredibly thorough. I finally understood the theory behind the algorithms I was using.",
      rating: 5
    },
    {
      name: "Priya Patel",
      role: "AI Researcher at OpenAI",
      image: "👩‍🎓",
      quote: "The cutting-edge content on transformers and generative AI helped me land my dream job in AI research.",
      rating: 5
    },
    {
      name: "David Kim",
      role: "Startup Founder",
      image: "👨‍💼",
      quote: "I built my entire AI startup based on the knowledge I gained here. The practical focus was exactly what I needed.",
      rating: 5
    }
  ]

  const stats = [
    { number: "50,000+", label: "Students Enrolled", icon: "👥" },
    { number: "95%", label: "Job Placement Rate", icon: "💼" },
    { number: "4.9/5", label: "Average Rating", icon: "⭐" },
    { number: "200+", label: "Hours of Content", icon: "📚" }
  ]

  const faqs = [
    {
      id: "prerequisites",
      question: "What prerequisites do I need to start?",
      answer: "No specific prerequisites are required! We start with mathematical foundations and programming basics. However, basic familiarity with high school mathematics and any programming language will help you progress faster."
    },
    {
      id: "time-commitment",
      question: "How much time should I dedicate to learning?",
      answer: "We recommend 5-10 hours per week for optimal progress. The course is self-paced, so you can adjust based on your schedule. Most students complete the full curriculum in 6-12 months."
    },
    {
      id: "job-prospects",
      question: "What job opportunities will this prepare me for?",
      answer: "Graduates work as ML Engineers, Data Scientists, AI Researchers, Computer Vision Engineers, NLP Engineers, and AI Product Managers at companies like Google, Microsoft, Tesla, and many startups."
    },
    {
      id: "practical-projects",
      question: "Will I work on real projects?",
      answer: "Absolutely! You'll build projects like image classifiers, chatbots, recommendation systems, and more. Each project uses real datasets and follows industry best practices."
    },
    {
      id: "support",
      question: "What kind of support do I get?",
      answer: "You'll have access to our community forum, weekly office hours with instructors, peer study groups, and one-on-one mentoring sessions for career guidance."
    },
    {
      id: "updates",
      question: "How do you keep content current?",
      answer: "Our curriculum is updated quarterly with the latest research and industry trends. You'll learn about cutting-edge developments like GPT models, diffusion models, and emerging AI techniques."
    }
  ]

  const companies = [
    { name: "Google", logo: "🔍" },
    { name: "Microsoft", logo: "🪟" },
    { name: "OpenAI", logo: "🤖" },
    { name: "Tesla", logo: "🚗" },
    { name: "Netflix", logo: "📺" },
    { name: "Meta", logo: "📘" },
    { name: "Amazon", logo: "📦" },
    { name: "Apple", logo: "🍎" }
  ]

  const handleLearningPathStart = (path: typeof learningPaths[0]) => {
    setSelectedLearningPath(path.id)
    setActiveSection(path.startSection)
  }

  const handleNewsletterSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    // Handle newsletter subscription
    alert('Thank you for subscribing! You\'ll receive our weekly AI insights.')
  }

  return (
    <main className="min-h-screen">
      <Navigation activeSection={activeSection} setActiveSection={setActiveSection} />
      
      {activeSection === 'home' && (
        <>
          <Hero />
          <CourseOverview />
          
          {/* Features Section */}
          <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gray-50">
            <div className="max-w-7xl mx-auto">
              <div className="text-center mb-16">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">
                  Why Choose Our <span className="gradient-text">ML Course</span>?
                </h2>
                <p className="text-lg text-gray-600 max-w-3xl mx-auto">
                  Our comprehensive approach combines theory, practice, and real-world application 
                  to give you the skills employers are looking for.
                </p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {features.map((feature, index) => (
                  <div key={index} className="bg-white rounded-xl shadow-lg p-6 card-hover">
                    <div className="text-4xl mb-4">{feature.icon}</div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-3">{feature.title}</h3>
                    <p className="text-gray-600 mb-4">{feature.description}</p>
                    <p className="text-sm text-gray-500">{feature.details}</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Learning Paths */}
          <section className="py-20 px-4 sm:px-6 lg:px-8">
            <div className="max-w-7xl mx-auto">
              <div className="text-center mb-16">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">
                  Choose Your <span className="gradient-text">Learning Path</span>
                </h2>
                <p className="text-lg text-gray-600 max-w-3xl mx-auto">
                  Whether you're just starting out or looking to advance your career, 
                  we have a path tailored to your experience level and goals.
                </p>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {learningPaths.map((path, index) => (
                  <div key={index} className={`rounded-xl border-2 p-6 ${path.color} ${selectedLearningPath === path.id ? 'ring-4 ring-purple-300' : ''}`}>
                    <div className="text-center mb-6">
                      <h3 className="text-2xl font-bold mb-2">{path.title}</h3>
                      <div className="text-sm font-medium mb-2">{path.duration}</div>
                      <p className="text-sm">{path.description}</p>
                    </div>
                    
                    <div className="space-y-3 mb-6">
                      <h4 className="font-semibold">What you'll learn:</h4>
                      <ul className="space-y-2">
                        {path.modules.map((module, moduleIndex) => (
                          <li key={moduleIndex} className="flex items-center text-sm">
                            <span className="w-2 h-2 bg-current rounded-full mr-3"></span>
                            {module}
                          </li>
                        ))}
                      </ul>
                    </div>
                    
                    <div className="border-t border-current/20 pt-4 mb-6">
                      <h4 className="font-semibold mb-2">Outcome:</h4>
                      <p className="text-sm">{path.outcome}</p>
                    </div>
                    
                    <button 
                      onClick={() => handleLearningPathStart(path)}
                      className="w-full px-6 py-3 bg-white text-gray-900 rounded-lg font-medium hover:bg-gray-50 transition-colors"
                    >
                      Start {path.title}
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <InteractiveDemo />

          {/* Statistics */}
          <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-purple-600 to-blue-600">
            <div className="max-w-7xl mx-auto">
              <div className="text-center mb-16">
                <h2 className="text-3xl font-bold text-white mb-4">
                  Join Thousands of Successful <span className="text-yellow-300">ML Engineers</span>
                </h2>
                <p className="text-lg text-purple-100 max-w-3xl mx-auto">
                  Our graduates are working at top tech companies and building the future of AI
                </p>
              </div>
              
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
                {stats.map((stat, index) => (
                  <div key={index} className="text-center">
                    <div className="text-4xl mb-2">{stat.icon}</div>
                    <div className="text-3xl font-bold text-white mb-2">{stat.number}</div>
                    <div className="text-purple-100">{stat.label}</div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Testimonials */}
          <section className="py-20 px-4 sm:px-6 lg:px-8">
            <div className="max-w-7xl mx-auto">
              <div className="text-center mb-16">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">
                  What Our <span className="gradient-text">Students Say</span>
                </h2>
                <p className="text-lg text-gray-600 max-w-3xl mx-auto">
                  Hear from graduates who've transformed their careers with our ML course
                </p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {testimonials.map((testimonial, index) => (
                  <div key={index} className="bg-white rounded-xl shadow-lg p-6">
                    <div className="flex items-center mb-4">
                      <div className="text-4xl mr-4">{testimonial.image}</div>
                      <div>
                        <h4 className="font-semibold text-gray-900">{testimonial.name}</h4>
                        <p className="text-sm text-gray-600">{testimonial.role}</p>
                      </div>
                      <div className="ml-auto">
                        <div className="flex text-yellow-400">
                          {[...Array(testimonial.rating)].map((_, i) => (
                            <span key={i}>⭐</span>
                          ))}
                        </div>
                      </div>
                    </div>
                    <p className="text-gray-700 italic">"{testimonial.quote}"</p>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Companies */}
          <section className="py-16 px-4 sm:px-6 lg:px-8 bg-gray-50">
            <div className="max-w-7xl mx-auto">
              <div className="text-center mb-12">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  Our Graduates Work At
                </h2>
                <p className="text-gray-600">
                  Join alumni at the world's leading technology companies
                </p>
              </div>
              
              <div className="grid grid-cols-4 lg:grid-cols-8 gap-8 items-center">
                {companies.map((company, index) => (
                  <div key={index} className="text-center">
                    <div className="text-3xl mb-2">{company.logo}</div>
                    <div className="text-sm text-gray-600">{company.name}</div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* FAQ Section */}
          <section className="py-20 px-4 sm:px-6 lg:px-8">
            <div className="max-w-4xl mx-auto">
              <div className="text-center mb-16">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">
                  Frequently Asked <span className="gradient-text">Questions</span>
                </h2>
                <p className="text-lg text-gray-600">
                  Get answers to common questions about our machine learning course
                </p>
              </div>
              
              <div className="space-y-4">
                {faqs.map((faq) => (
                  <div key={faq.id} className="bg-white rounded-lg shadow-md">
                    <button
                      className="w-full px-6 py-4 text-left flex items-center justify-between hover:bg-gray-50 transition-colors"
                      onClick={() => setExpandedFaq(expandedFaq === faq.id ? null : faq.id)}
                    >
                      <h3 className="font-semibold text-gray-900">{faq.question}</h3>
                      <span className="text-2xl text-gray-400">
                        {expandedFaq === faq.id ? '−' : '+'}
                      </span>
                    </button>
                    {expandedFaq === faq.id && (
                      <div className="px-6 pb-4">
                        <p className="text-gray-600">{faq.answer}</p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Newsletter Signup */}
          <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-r from-blue-600 to-purple-600">
            <div className="max-w-4xl mx-auto text-center">
              <h2 className="text-3xl font-bold text-white mb-4">
                Stay Updated with <span className="text-yellow-300">AI Trends</span>
              </h2>
              <p className="text-lg text-blue-100 mb-8 max-w-2xl mx-auto">
                Get weekly insights on the latest developments in machine learning, 
                AI research breakthroughs, and career opportunities.
              </p>
              
              <form onSubmit={handleNewsletterSubmit} className="flex flex-col sm:flex-row gap-4 max-w-md mx-auto">
                <input
                  type="email"
                  placeholder="Enter your email"
                  required
                  className="flex-1 px-4 py-3 rounded-lg border-0 focus:ring-2 focus:ring-yellow-300 focus:outline-none"
                />
                <button 
                  type="submit"
                  className="px-6 py-3 bg-yellow-400 text-gray-900 rounded-lg font-medium hover:bg-yellow-300 transition-colors"
                >
                  Subscribe
                </button>
              </form>
              
              <p className="text-sm text-blue-200 mt-4">
                Join 25,000+ ML enthusiasts. Unsubscribe anytime.
              </p>
            </div>
          </section>

          {/* Call to Action */}
          <section className="py-20 px-4 sm:px-6 lg:px-8">
            <div className="max-w-4xl mx-auto text-center">
              <h2 className="text-4xl font-bold text-gray-900 mb-6">
                Ready to Start Your <span className="gradient-text">ML Journey</span>?
              </h2>
              <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
                Join thousands of students who've transformed their careers with our 
                comprehensive machine learning course.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <button 
                  onClick={() => setActiveSection('mathematics')}
                  className="px-8 py-4 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 transition-colors text-lg"
                >
                  Start Learning Now
                </button>
                <button 
                  onClick={() => setActiveSection('machine-learning')}
                  className="px-8 py-4 border-2 border-purple-600 text-purple-600 rounded-lg font-medium hover:bg-purple-50 transition-colors text-lg"
                >
                  Explore Curriculum
                </button>
              </div>
              
              <div className="mt-8 flex items-center justify-center space-x-6 text-sm text-gray-500">
                <div className="flex items-center">
                  <span className="text-green-500 mr-2">✓</span>
                  No credit card required
                </div>
                <div className="flex items-center">
                  <span className="text-green-500 mr-2">✓</span>
                  Start immediately
                </div>
                <div className="flex items-center">
                  <span className="text-green-500 mr-2">✓</span>
                  30-day money back guarantee
                </div>
              </div>
            </div>
          </section>
        </>
      )}
      
      {activeSection === 'mathematics' && <MathPrerequisites />}
      {activeSection === 'history' && <HistoryTimeline />}
      {activeSection === 'machine-learning' && <MachineLearningContent />}
      {activeSection === 'deep-learning' && <DeepLearningContent />}
      {activeSection === 'quizzes' && <QuizSystem />}
      
      <Footer />
    </main>
  )
} 