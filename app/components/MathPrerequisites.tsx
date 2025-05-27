'use client'

import { useState } from 'react'

export default function MathPrerequisites() {
  const [activeSection, setActiveSection] = useState('foundations')
  const [expandedTopic, setExpandedTopic] = useState<string | null>(null)

  const mathSections = {
    foundations: {
      title: "Mathematical Foundations",
      description: "Essential mathematical concepts and notation",
      topics: [
        {
          title: "Sets and Logic",
          description: "Fundamental building blocks of mathematics",
          concepts: [
            { 
              name: "Set Theory", 
              formula: "A = {1, 2, 3}, B = {2, 3, 4}", 
              description: "Collections of distinct objects",
              details: "Sets form the foundation of mathematics. Key operations include union (A ∪ B), intersection (A ∩ B), and complement (A^c). Understanding sets is crucial for probability theory and data structures in ML."
            },
            { 
              name: "Logic Operations", 
              formula: "P ∧ Q, P ∨ Q, ¬P, P → Q", 
              description: "AND, OR, NOT, and implication operations",
              details: "Boolean logic is essential for understanding conditional statements, decision trees, and logical reasoning in AI systems."
            },
            { 
              name: "Quantifiers", 
              formula: "∀x ∈ S, P(x) and ∃x ∈ S, P(x)", 
              description: "Universal (for all) and existential (there exists) quantifiers",
              details: "Quantifiers help express mathematical statements precisely and are fundamental in formal logic and proof techniques."
            },
            { 
              name: "Functions", 
              formula: "f: X → Y, f(x) = y", 
              description: "Mappings between sets",
              details: "Functions are the mathematical foundation of neural networks, where each layer applies a function to transform inputs to outputs."
            }
          ],
          applications: ["Boolean algebra in neural networks", "Set operations in data preprocessing", "Logical reasoning in AI"]
        },
        {
          title: "Number Systems",
          description: "Different types of numbers and their properties",
          concepts: [
            { 
              name: "Natural Numbers", 
              formula: "ℕ = {1, 2, 3, 4, ...}", 
              description: "Counting numbers",
              details: "Used for indexing, counting samples, and discrete variables in machine learning."
            },
            { 
              name: "Integers", 
              formula: "ℤ = {..., -2, -1, 0, 1, 2, ...}", 
              description: "Whole numbers including negatives",
              details: "Important for signed representations and difference calculations in algorithms."
            },
            { 
              name: "Rational Numbers", 
              formula: "ℚ = {p/q : p, q ∈ ℤ, q ≠ 0}", 
              description: "Fractions and ratios",
              details: "Used in probability calculations and normalized values in ML."
            },
            { 
              name: "Real Numbers", 
              formula: "ℝ = ℚ ∪ {irrational numbers}", 
              description: "All numbers on the number line",
              details: "The primary number system for continuous variables, weights, and activations in neural networks."
            },
            { 
              name: "Complex Numbers", 
              formula: "ℂ = {a + bi : a, b ∈ ℝ, i² = -1}", 
              description: "Numbers with real and imaginary parts",
              details: "Used in signal processing, Fourier transforms, and some advanced ML techniques."
            }
          ],
          applications: ["Data type selection", "Numerical precision", "Complex-valued neural networks"]
        },
        {
          title: "Basic Algebra",
          description: "Algebraic manipulation and equation solving",
          concepts: [
            { 
              name: "Polynomial Operations", 
              formula: "P(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀", 
              description: "Operations with polynomial expressions",
              details: "Polynomials appear in activation functions, loss functions, and approximation theory in ML."
            },
            { 
              name: "Exponentials", 
              formula: "aˣ, e^x, log_a(x), ln(x)", 
              description: "Exponential and logarithmic functions",
              details: "Fundamental for activation functions (sigmoid, softmax), loss functions (cross-entropy), and growth models."
            },
            { 
              name: "Inequalities", 
              formula: "a < b, a ≤ b, |x| ≤ M", 
              description: "Ordering relationships and bounds",
              details: "Essential for optimization constraints, convergence analysis, and regularization techniques."
            },
            { 
              name: "Summation Notation", 
              formula: "∑ᵢ₌₁ⁿ aᵢ, ∏ᵢ₌₁ⁿ aᵢ", 
              description: "Compact notation for sums and products",
              details: "Ubiquitous in ML for expressing loss functions, gradients, and statistical measures."
            }
          ],
          applications: ["Loss function formulation", "Activation functions", "Optimization constraints"]
        }
      ]
    },
    linearAlgebra: {
      title: "Linear Algebra",
      description: "Vectors, matrices, and linear transformations",
      topics: [
        {
          title: "Vectors",
          description: "Mathematical objects with magnitude and direction",
          concepts: [
            { 
              name: "Vector Definition", 
              formula: "v = [v₁, v₂, ..., vₙ] ∈ ℝⁿ", 
              description: "Ordered lists of numbers",
              details: "Vectors represent data points, features, weights, and gradients in ML. They can be thought of as arrows in n-dimensional space."
            },
            { 
              name: "Vector Addition", 
              formula: "u + v = [u₁+v₁, u₂+v₂, ..., uₙ+vₙ]", 
              description: "Component-wise addition",
              details: "Used in gradient updates, combining features, and linear combinations in neural networks."
            },
            { 
              name: "Scalar Multiplication", 
              formula: "αv = [αv₁, αv₂, ..., αvₙ]", 
              description: "Scaling a vector by a constant",
              details: "Essential for learning rates, weight scaling, and normalization operations."
            },
            { 
              name: "Dot Product", 
              formula: "u·v = ∑ᵢ₌₁ⁿ uᵢvᵢ = ||u||||v||cos(θ)", 
              description: "Measures similarity and projection",
              details: "Fundamental operation in neural networks, attention mechanisms, and similarity measures."
            },
            { 
              name: "Vector Norms", 
              formula: "||v||₂ = √(∑ᵢ₌₁ⁿ vᵢ²), ||v||₁ = ∑ᵢ₌₁ⁿ |vᵢ|", 
              description: "Measures of vector magnitude",
              details: "Used in regularization (L1, L2), gradient clipping, and normalization techniques."
            }
          ],
          applications: ["Feature vectors", "Weight vectors", "Gradient computation", "Attention scores"]
        },
        {
          title: "Matrices",
          description: "Rectangular arrays of numbers with rich algebraic structure",
          concepts: [
            { 
              name: "Matrix Definition", 
              formula: "A ∈ ℝᵐˣⁿ, A = [aᵢⱼ]", 
              description: "2D arrays of numbers",
              details: "Matrices represent linear transformations, weight matrices in neural networks, and data matrices where rows are samples and columns are features."
            },
            { 
              name: "Matrix Multiplication", 
              formula: "(AB)ᵢⱼ = ∑ₖ₌₁ⁿ AᵢₖBₖⱼ", 
              description: "Composition of linear transformations",
              details: "The core operation in neural networks, representing how information flows through layers."
            },
            { 
              name: "Matrix Transpose", 
              formula: "(Aᵀ)ᵢⱼ = Aⱼᵢ", 
              description: "Flipping rows and columns",
              details: "Used in backpropagation, symmetric matrices, and changing data layout for efficient computation."
            },
            { 
              name: "Matrix Inverse", 
              formula: "A⁻¹A = AA⁻¹ = I", 
              description: "Undoing a linear transformation",
              details: "Important for solving linear systems, though rarely computed directly in ML due to numerical stability."
            },
            { 
              name: "Determinant", 
              formula: "det(A) = ∑σ sgn(σ)∏ᵢ₌₁ⁿ aᵢ,σ(ᵢ)", 
              description: "Scalar measure of matrix properties",
              details: "Indicates invertibility, volume scaling, and appears in probability density functions."
            }
          ],
          applications: ["Weight matrices", "Data transformation", "Covariance matrices", "Linear layers"]
        },
        {
          title: "Eigenvalues and Eigenvectors",
          description: "Special vectors that reveal matrix structure",
          concepts: [
            { 
              name: "Eigenvalue Equation", 
              formula: "Av = λv, λ ∈ ℝ, v ≠ 0", 
              description: "Vectors that don't change direction under transformation",
              details: "Eigenvectors reveal the principal directions of a linear transformation, crucial for PCA and understanding matrix behavior."
            },
            { 
              name: "Characteristic Polynomial", 
              formula: "det(A - λI) = 0", 
              description: "Polynomial whose roots are eigenvalues",
              details: "Method for computing eigenvalues, though iterative methods are preferred for large matrices."
            },
            { 
              name: "Eigendecomposition", 
              formula: "A = QΛQᵀ", 
              description: "Decomposing matrix into eigencomponents",
              details: "Fundamental for PCA, spectral analysis, and understanding quadratic forms in optimization."
            },
            { 
              name: "Singular Value Decomposition", 
              formula: "A = UΣVᵀ", 
              description: "Generalization to non-square matrices",
              details: "More general than eigendecomposition, used in dimensionality reduction, matrix completion, and neural network analysis."
            }
          ],
          applications: ["Principal Component Analysis", "Spectral clustering", "Matrix factorization", "Stability analysis"]
        }
      ]
    },
    calculus: {
      title: "Calculus",
      description: "Derivatives, integrals, and optimization",
      topics: [
        {
          title: "Limits and Continuity",
          description: "Foundation of calculus and analysis",
          concepts: [
            { 
              name: "Limit Definition", 
              formula: "lim_{x→a} f(x) = L", 
              description: "Behavior of functions as input approaches a value",
              details: "Limits define derivatives and integrals, and are crucial for understanding convergence in optimization algorithms."
            },
            { 
              name: "Continuity", 
              formula: "lim_{x→a} f(x) = f(a)", 
              description: "Functions without jumps or breaks",
              details: "Continuous functions are easier to optimize and have better theoretical properties for machine learning."
            },
            { 
              name: "Epsilon-Delta Definition", 
              formula: "∀ε>0, ∃δ>0: |x-a|<δ ⟹ |f(x)-L|<ε", 
              description: "Rigorous definition of limits",
              details: "Provides the mathematical foundation for proving convergence of algorithms and stability of solutions."
            }
          ],
          applications: ["Convergence analysis", "Stability theory", "Approximation theory"]
        },
        {
          title: "Derivatives",
          description: "Rates of change and local linear approximation",
          concepts: [
            { 
              name: "Derivative Definition", 
              formula: "f'(x) = lim_{h→0} [f(x+h) - f(x)]/h", 
              description: "Instantaneous rate of change",
              details: "Derivatives measure how sensitive a function is to changes in input, forming the basis of gradient-based optimization."
            },
            { 
              name: "Chain Rule", 
              formula: "(f∘g)'(x) = f'(g(x))·g'(x)", 
              description: "Derivative of composite functions",
              details: "The mathematical foundation of backpropagation algorithm in neural networks."
            },
            { 
              name: "Product Rule", 
              formula: "(fg)' = f'g + fg'", 
              description: "Derivative of products",
              details: "Used when differentiating complex expressions involving products of functions."
            },
            { 
              name: "Quotient Rule", 
              formula: "(f/g)' = (f'g - fg')/g²", 
              description: "Derivative of quotients",
              details: "Important for functions involving ratios, such as some activation functions and loss functions."
            },
            { 
              name: "Partial Derivatives", 
              formula: "∂f/∂x = lim_{h→0} [f(x+h,y) - f(x,y)]/h", 
              description: "Derivatives with respect to one variable",
              details: "Essential for multivariable optimization and computing gradients in machine learning."
            }
          ],
          applications: ["Gradient computation", "Backpropagation", "Sensitivity analysis", "Optimization"]
        },
        {
          title: "Multivariable Calculus",
          description: "Calculus in higher dimensions",
          concepts: [
            { 
              name: "Gradient Vector", 
              formula: "∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]", 
              description: "Vector of all partial derivatives",
              details: "Points in the direction of steepest increase, used in gradient descent and other optimization algorithms."
            },
            { 
              name: "Hessian Matrix", 
              formula: "H = [∂²f/∂xᵢ∂xⱼ]", 
              description: "Matrix of second partial derivatives",
              details: "Describes the curvature of functions, used in second-order optimization methods like Newton's method."
            },
            { 
              name: "Jacobian Matrix", 
              formula: "J = [∂fᵢ/∂xⱼ]", 
              description: "Matrix of partial derivatives for vector functions",
              details: "Essential for backpropagation through vector-valued functions and understanding local linear approximations."
            },
            { 
              name: "Directional Derivative", 
              formula: "D_u f = ∇f · u", 
              description: "Rate of change in a specific direction",
              details: "Generalizes the concept of derivative to arbitrary directions, important for understanding optimization landscapes."
            }
          ],
          applications: ["Gradient descent", "Newton's method", "Backpropagation", "Optimization theory"]
        },
        {
          title: "Optimization",
          description: "Finding extrema of functions",
          concepts: [
            { 
              name: "Critical Points", 
              formula: "∇f(x*) = 0", 
              description: "Points where gradient is zero",
              details: "Candidates for local minima, maxima, or saddle points in optimization problems."
            },
            { 
              name: "Second Derivative Test", 
              formula: "f''(x) > 0 ⟹ local min, f''(x) < 0 ⟹ local max", 
              description: "Classifying critical points",
              details: "Helps determine the nature of critical points, extended to multiple dimensions using the Hessian."
            },
            { 
              name: "Lagrange Multipliers", 
              formula: "∇f = λ∇g at constrained optimum", 
              description: "Optimization with equality constraints",
              details: "Method for solving constrained optimization problems, important in support vector machines and other ML algorithms."
            },
            { 
              name: "Convexity", 
              formula: "f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)", 
              description: "Functions with unique global minima",
              details: "Convex functions have nice optimization properties and guarantee global optimality for gradient-based methods."
            }
          ],
          applications: ["Loss minimization", "Constrained optimization", "Convex optimization", "Regularization"]
        }
      ]
    },
    probability: {
      title: "Probability & Statistics",
      description: "Uncertainty, randomness, and statistical inference",
      topics: [
        {
          title: "Probability Fundamentals",
          description: "Basic concepts of probability theory",
          concepts: [
            { 
              name: "Sample Space", 
              formula: "Ω = {all possible outcomes}", 
              description: "Set of all possible outcomes",
              details: "The foundation of probability theory, defining the universe of possible events in a random experiment."
            },
            { 
              name: "Probability Axioms", 
              formula: "P(Ω) = 1, P(A) ≥ 0, P(A∪B) = P(A)+P(B) if A∩B=∅", 
              description: "Fundamental rules of probability",
              details: "These axioms ensure probability measures are mathematically consistent and form the basis for all probability calculations."
            },
            { 
              name: "Conditional Probability", 
              formula: "P(A|B) = P(A∩B)/P(B)", 
              description: "Probability given additional information",
              details: "Fundamental for Bayesian reasoning, decision trees, and understanding dependencies between variables."
            },
            { 
              name: "Independence", 
              formula: "P(A∩B) = P(A)P(B)", 
              description: "Events that don't influence each other",
              details: "Important assumption in many ML models, though real-world data often violates independence."
            },
            { 
              name: "Bayes' Theorem", 
              formula: "P(A|B) = P(B|A)P(A)/P(B)", 
              description: "Updating beliefs with new evidence",
              details: "Foundation of Bayesian machine learning, spam filtering, and probabilistic reasoning."
            }
          ],
          applications: ["Bayesian inference", "Naive Bayes classifier", "Probabilistic models", "Decision theory"]
        },
        {
          title: "Random Variables",
          description: "Functions that assign numbers to random outcomes",
          concepts: [
            { 
              name: "Discrete Random Variables", 
              formula: "P(X = x) = p(x), ∑_x p(x) = 1", 
              description: "Variables with countable outcomes",
              details: "Used for categorical data, classification problems, and discrete choice models."
            },
            { 
              name: "Continuous Random Variables", 
              formula: "P(a ≤ X ≤ b) = ∫_a^b f(x)dx", 
              description: "Variables with uncountable outcomes",
              details: "Used for continuous measurements, regression targets, and continuous probability distributions."
            },
            { 
              name: "Expectation", 
              formula: "E[X] = ∑_x x·p(x) or ∫ x·f(x)dx", 
              description: "Average value of a random variable",
              details: "Central concept in statistics, used in loss functions, risk assessment, and decision theory."
            },
            { 
              name: "Variance", 
              formula: "Var(X) = E[(X-μ)²] = E[X²] - (E[X])²", 
              description: "Measure of spread around the mean",
              details: "Quantifies uncertainty and variability, important for understanding model reliability and confidence intervals."
            }
          ],
          applications: ["Feature distributions", "Loss function design", "Uncertainty quantification", "Risk assessment"]
        },
        {
          title: "Common Distributions",
          description: "Important probability distributions in ML",
          concepts: [
            { 
              name: "Bernoulli Distribution", 
              formula: "P(X=1) = p, P(X=0) = 1-p", 
              description: "Single binary trial",
              details: "Models binary outcomes like coin flips, binary classification, and success/failure events."
            },
            { 
              name: "Binomial Distribution", 
              formula: "P(X=k) = C(n,k)p^k(1-p)^(n-k)", 
              description: "Number of successes in n trials",
              details: "Models repeated binary trials, useful for understanding sampling and classification accuracy."
            },
            { 
              name: "Normal Distribution", 
              formula: "f(x) = (1/√(2πσ²))e^(-(x-μ)²/(2σ²))", 
              description: "Bell-shaped continuous distribution",
              details: "Central to statistics due to Central Limit Theorem, used in many ML algorithms and error models."
            },
            { 
              name: "Exponential Distribution", 
              formula: "f(x) = λe^(-λx) for x ≥ 0", 
              description: "Time between events in Poisson process",
              details: "Models waiting times, survival analysis, and appears in some activation functions."
            },
            { 
              name: "Categorical Distribution", 
              formula: "P(X=k) = p_k, ∑_k p_k = 1", 
              description: "Generalization of Bernoulli to multiple categories",
              details: "Foundation of multiclass classification and the softmax function in neural networks."
            }
          ],
          applications: ["Classification models", "Generative models", "Bayesian priors", "Error modeling"]
        }
      ]
    },
    informationTheory: {
      title: "Information Theory",
      description: "Quantifying information and uncertainty",
      topics: [
        {
          title: "Information Measures",
          description: "Quantifying information content and uncertainty",
          concepts: [
            { 
              name: "Self-Information", 
              formula: "I(x) = -log₂(P(x))", 
              description: "Information content of a single event",
              details: "Rare events carry more information than common events. Forms the basis for entropy and other information measures."
            },
            { 
              name: "Entropy", 
              formula: "H(X) = -∑_x P(x)log₂P(x) = E[-log₂P(X)]", 
              description: "Expected information content",
              details: "Measures uncertainty in a random variable. Higher entropy means more uncertainty and more information needed to describe the variable."
            },
            { 
              name: "Cross-Entropy", 
              formula: "H(p,q) = -∑_x p(x)log q(x)", 
              description: "Information needed when using wrong distribution",
              details: "Fundamental loss function in classification, measures the difference between true and predicted distributions."
            },
            { 
              name: "KL Divergence", 
              formula: "D_KL(P||Q) = ∑_x p(x)log(p(x)/q(x))", 
              description: "Relative entropy between distributions",
              details: "Measures how much one distribution differs from another, used in variational inference and model comparison."
            },
            { 
              name: "Mutual Information", 
              formula: "I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)", 
              description: "Shared information between variables",
              details: "Measures dependence between variables, used in feature selection and understanding relationships in data."
            }
          ],
          applications: ["Loss functions", "Feature selection", "Model compression", "Variational inference"]
        }
      ]
    }
  }

  const sectionOrder = ['foundations', 'linearAlgebra', 'calculus', 'probability', 'informationTheory']

  return (
    <section className="pt-20 pb-16 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Mathematical <span className="gradient-text">Foundations</span>
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Master the essential mathematics behind machine learning and deep learning algorithms, 
            from basic foundations to advanced concepts.
          </p>
        </div>

        {/* Navigation */}
        <div className="mb-8">
          <div className="flex flex-wrap justify-center gap-2 mb-6">
            {sectionOrder.map((sectionKey) => (
              <button
                key={sectionKey}
                onClick={() => setActiveSection(sectionKey)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  activeSection === sectionKey
                    ? 'bg-purple-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {mathSections[sectionKey as keyof typeof mathSections].title}
              </button>
            ))}
          </div>
        </div>

        {/* Active Section Content */}
        <div className="space-y-8">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-2">
              {mathSections[activeSection as keyof typeof mathSections].title}
            </h2>
            <p className="text-lg text-gray-600">
              {mathSections[activeSection as keyof typeof mathSections].description}
            </p>
          </div>

          {mathSections[activeSection as keyof typeof mathSections].topics.map((topic, topicIndex) => (
            <div key={topicIndex} className="bg-white rounded-xl shadow-lg overflow-hidden">
              <div 
                className="p-6 cursor-pointer hover:bg-gray-50 transition-colors"
                onClick={() => setExpandedTopic(expandedTopic === `${activeSection}-${topicIndex}` ? null : `${activeSection}-${topicIndex}`)}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-xl font-bold text-gray-900 mb-2">{topic.title}</h3>
                    <p className="text-gray-600">{topic.description}</p>
                  </div>
                  <div className="text-2xl text-gray-400">
                    {expandedTopic === `${activeSection}-${topicIndex}` ? '−' : '+'}
                  </div>
                </div>
              </div>

              {expandedTopic === `${activeSection}-${topicIndex}` && (
                <div className="px-6 pb-6">
                  {/* Key Concepts */}
                  <div className="mb-6">
                    <h4 className="text-lg font-semibold text-gray-900 mb-4">Key Concepts</h4>
                    <div className="space-y-4">
                      {topic.concepts.map((concept, conceptIndex) => (
                        <div key={conceptIndex} className="border border-gray-200 rounded-lg p-4 hover:border-purple-300 transition-colors">
                          <h5 className="font-semibold text-gray-900 mb-2">{concept.name}</h5>
                          <div className="bg-gray-50 rounded p-3 mb-3 font-mono text-sm overflow-x-auto">
                            {concept.formula}
                          </div>
                          <p className="text-sm text-gray-600 mb-2">{concept.description}</p>
                          <p className="text-xs text-gray-500">{concept.details}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Applications */}
                  <div>
                    <h4 className="text-lg font-semibold text-gray-900 mb-4">ML/DL Applications</h4>
                    <div className="flex flex-wrap gap-2">
                      {topic.applications.map((app, appIndex) => (
                        <span key={appIndex} className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm">
                          {app}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Learning Path */}
        <div className="mt-16 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Recommended Learning Path</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {sectionOrder.map((sectionKey, index) => (
              <div key={sectionKey} className="bg-white rounded-lg p-4 text-center">
                <div className={`w-12 h-12 text-white rounded-full flex items-center justify-center mx-auto mb-3 text-xl font-bold ${
                  index === 0 ? 'bg-red-500' :
                  index === 1 ? 'bg-blue-500' :
                  index === 2 ? 'bg-green-500' :
                  index === 3 ? 'bg-purple-500' : 'bg-orange-500'
                }`}>
                  {index + 1}
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">
                  {mathSections[sectionKey as keyof typeof mathSections].title}
                </h3>
                <p className="text-sm text-gray-600">
                  {index === 0 && "Start with basic mathematical concepts"}
                  {index === 1 && "Learn vectors and matrices"}
                  {index === 2 && "Master derivatives and optimization"}
                  {index === 3 && "Understand uncertainty and inference"}
                  {index === 4 && "Quantify information and entropy"}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* Interactive Examples */}
        <div className="mt-12 bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Interactive Examples & Practice</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="border border-gray-200 rounded-lg p-4 hover:border-purple-300 transition-colors">
              <h3 className="font-semibold text-gray-900 mb-2">🧮 Matrix Operations</h3>
              <p className="text-gray-600 text-sm mb-3">Practice matrix multiplication, eigenvalues, and decompositions</p>
              <button className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors">
                Start Practice
              </button>
            </div>
            <div className="border border-gray-200 rounded-lg p-4 hover:border-purple-300 transition-colors">
              <h3 className="font-semibold text-gray-900 mb-2">📈 Gradient Computation</h3>
              <p className="text-gray-600 text-sm mb-3">Calculate gradients for common ML functions and neural networks</p>
              <button className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors">
                Start Practice
              </button>
            </div>
            <div className="border border-gray-200 rounded-lg p-4 hover:border-purple-300 transition-colors">
              <h3 className="font-semibold text-gray-900 mb-2">🎲 Probability Distributions</h3>
              <p className="text-gray-600 text-sm mb-3">Explore different distributions and their properties</p>
              <button className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 transition-colors">
                Start Practice
              </button>
            </div>
            <div className="border border-gray-200 rounded-lg p-4 hover:border-purple-300 transition-colors">
              <h3 className="font-semibold text-gray-900 mb-2">🔍 Optimization Problems</h3>
              <p className="text-gray-600 text-sm mb-3">Solve optimization problems using various methods</p>
              <button className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors">
                Start Practice
              </button>
            </div>
            <div className="border border-gray-200 rounded-lg p-4 hover:border-purple-300 transition-colors">
              <h3 className="font-semibold text-gray-900 mb-2">📊 Information Theory</h3>
              <p className="text-gray-600 text-sm mb-3">Calculate entropy, KL divergence, and mutual information</p>
              <button className="px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-700 transition-colors">
                Start Practice
              </button>
            </div>
            <div className="border border-gray-200 rounded-lg p-4 hover:border-purple-300 transition-colors">
              <h3 className="font-semibold text-gray-900 mb-2">🧠 ML Applications</h3>
              <p className="text-gray-600 text-sm mb-3">See how math concepts apply to real ML problems</p>
              <button className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors">
                Start Practice
              </button>
            </div>
          </div>
        </div>

        {/* Quick Reference */}
        <div className="mt-12 bg-gray-50 rounded-xl p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Quick Reference</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-white rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-3">Common Derivatives</h3>
              <div className="space-y-2 text-sm font-mono">
                <div>d/dx[x^n] = nx^(n-1)</div>
                <div>d/dx[e^x] = e^x</div>
                <div>d/dx[ln(x)] = 1/x</div>
                <div>d/dx[sin(x)] = cos(x)</div>
                <div>d/dx[cos(x)] = -sin(x)</div>
              </div>
            </div>
            <div className="bg-white rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-3">Matrix Properties</h3>
              <div className="space-y-2 text-sm font-mono">
                <div>(AB)^T = B^T A^T</div>
                <div>(A^-1)^T = (A^T)^-1</div>
                <div>det(AB) = det(A)det(B)</div>
                <div>tr(A+B) = tr(A) + tr(B)</div>
                <div>rank(AB) ≤ min(rank(A), rank(B))</div>
              </div>
            </div>
            <div className="bg-white rounded-lg p-4">
              <h3 className="font-semibold text-gray-900 mb-3">Probability Rules</h3>
              <div className="space-y-2 text-sm font-mono">
                <div>P(A ∪ B) = P(A) + P(B) - P(A ∩ B)</div>
                <div>P(A^c) = 1 - P(A)</div>
                <div>P(A|B) = P(A ∩ B) / P(B)</div>
                <div>E[aX + b] = aE[X] + b</div>
                <div>Var(aX + b) = a²Var(X)</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
} 