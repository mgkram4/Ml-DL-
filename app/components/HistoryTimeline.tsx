'use client'

import { useState } from 'react'

export default function HistoryTimeline() {
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [selectedDecade, setSelectedDecade] = useState('all')
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null)

  const timelineEvents = [
    {
      id: "1847-boolean",
      year: "1847",
      title: "Boolean Algebra",
      description: "George Boole develops Boolean algebra, laying mathematical foundation for digital logic",
      category: "Mathematics",
      importance: "Foundation",
      keyFigures: ["George Boole"],
      details: "Boole's 'An Investigation of the Laws of Thought' introduced Boolean algebra, providing the mathematical foundation for all digital logic and computer science. This work established the logical operations (AND, OR, NOT) that would become fundamental to computing.",
      impact: "Created the mathematical framework that enables all digital computation and logical reasoning in machines.",
      technicalDetails: "Defined binary variables and logical operations that map perfectly to electrical circuits with on/off states."
    },
    {
      id: "1936-turing-machine",
      year: "1936",
      title: "Turing Machine Concept",
      description: "Alan Turing introduces the theoretical foundation of computation",
      category: "Computer Theory",
      importance: "Foundation",
      keyFigures: ["Alan Turing"],
      details: "Turing's paper 'On Computable Numbers' introduced the concept of a universal computing machine, now known as a Turing machine. This theoretical framework defined the limits of mechanical computation.",
      impact: "Established the theoretical foundations of computer science and computational complexity.",
      technicalDetails: "Defined a simple abstract machine with tape, states, and transition rules that could compute any computable function."
    },
    {
      id: "1943-mcculloch",
      year: "1943",
      title: "McCulloch-Pitts Neuron",
      description: "Warren McCulloch and Walter Pitts create the first mathematical model of a neural network",
      category: "Neural Networks",
      importance: "Foundation",
      keyFigures: ["Warren McCulloch", "Walter Pitts"],
      details: "This groundbreaking paper 'A Logical Calculus of the Ideas Immanent in Nervous Activity' introduced the first mathematical model of an artificial neuron. The McCulloch-Pitts neuron was a binary threshold unit that could perform logical operations, laying the theoretical foundation for all future neural network research.",
      impact: "Established the mathematical basis for artificial neural networks and computational neuroscience.",
      technicalDetails: "The model used binary inputs and outputs with weighted connections, introducing the concept of threshold activation."
    },
    {
      id: "1945-von-neumann",
      year: "1945",
      title: "Von Neumann Architecture",
      description: "John von Neumann describes stored-program computer architecture",
      category: "Computing",
      importance: "Foundation",
      keyFigures: ["John von Neumann", "Herman Goldstine"],
      details: "Von Neumann's 'First Draft of a Report on the EDVAC' described the architecture for stored-program computers, where both data and instructions are stored in memory.",
      impact: "Became the standard architecture for virtually all computers and enabled programmable machines.",
      technicalDetails: "Featured separate memory for data and instructions, with a central processing unit performing operations."
    },
    {
      id: "1945-eniac",
      year: "1945",
      title: "ENIAC Computer",
      description: "First general-purpose electronic digital computer, enabling complex calculations",
      category: "Computing",
      importance: "Foundation",
      keyFigures: ["John Mauchly", "J. Presper Eckert", "Betty Snyder", "Marlyn Wescoff", "Ruth Lichterman", "Betty Holberton", "Kay McNulty", "Fran Bilas"],
      details: "The Electronic Numerical Integrator and Computer (ENIAC) was one of the first general-purpose electronic digital computers. Weighing 30 tons and occupying 1,800 square feet, it could perform calculations thousands of times faster than manual computation. The programming was done by six women mathematicians who became the first computer programmers.",
      impact: "Demonstrated the potential of electronic computation for complex mathematical problems and established computer programming as a field.",
      technicalDetails: "Used 17,468 vacuum tubes, 7,200 crystal diodes, and consumed 150 kW of power."
    },
    {
      id: "1948-wiener",
      year: "1948",
      title: "Cybernetics",
      description: "Norbert Wiener establishes cybernetics, the study of communication and control",
      category: "AI Theory",
      importance: "Foundation",
      keyFigures: ["Norbert Wiener"],
      details: "Wiener's book 'Cybernetics: Or Control and Communication in the Animal and the Machine' established cybernetics as a field studying feedback, control, and communication in systems.",
      impact: "Influenced early AI research and established principles of feedback and control that remain important in AI.",
      technicalDetails: "Introduced concepts of feedback loops, homeostasis, and information theory applied to both biological and artificial systems."
    },
    {
      id: "1949-hebb",
      year: "1949",
      title: "Hebbian Learning",
      description: "Donald Hebb introduces the concept of synaptic plasticity in learning",
      category: "Neural Networks",
      importance: "Foundation",
      keyFigures: ["Donald Hebb"],
      details: "Hebb's book 'The Organization of Behavior' introduced the concept that neurons that fire together wire together, providing a biological basis for learning in neural networks.",
      impact: "Established the theoretical foundation for learning in neural networks and influenced decades of neuroscience and AI research.",
      technicalDetails: "Hebbian learning rule: when two neurons are simultaneously active, the connection between them is strengthened."
    },
    {
      id: "1950-turing",
      year: "1950",
      title: "Turing Test",
      description: "Alan Turing proposes the Turing Test as a measure of machine intelligence",
      category: "AI Theory",
      importance: "Foundation",
      keyFigures: ["Alan Turing"],
      details: "In his paper 'Computing Machinery and Intelligence,' Turing proposed the famous test where a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human would be considered artificial intelligence.",
      impact: "Established a benchmark for machine intelligence that remains influential today.",
      technicalDetails: "The test involves a human evaluator engaging in conversations with both a human and a machine, without knowing which is which."
    },
    {
      id: "1950-shannon",
      year: "1950",
      title: "Information Theory and Chess",
      description: "Claude Shannon publishes on chess programming and information theory applications",
      category: "Game AI",
      importance: "Innovation",
      keyFigures: ["Claude Shannon"],
      details: "Shannon's paper 'Programming a Computer for Playing Chess' outlined algorithms for computer chess and introduced minimax search with alpha-beta pruning.",
      impact: "Established the foundation for game-playing AI and search algorithms.",
      technicalDetails: "Introduced game tree search, evaluation functions, and pruning techniques still used in modern game AI."
    },
    {
      id: "1951-ferranti",
      year: "1951",
      title: "First Stored-Program Computer",
      description: "Ferranti Mark 1 becomes the first commercially available stored-program computer",
      category: "Computing",
      importance: "Milestone",
      keyFigures: ["Freddie Williams", "Tom Kilburn"],
      details: "The Ferranti Mark 1 was based on the Manchester Baby and became the first commercially available general-purpose stored-program computer, making computing accessible beyond research institutions.",
      impact: "Commercialized computing technology, making it available for broader research and applications.",
      technicalDetails: "Featured 4,050 vacuum tubes and could store 128 40-bit words in memory."
    },
    {
      id: "1952-checkers",
      year: "1952",
      title: "First Learning Program",
      description: "Arthur Samuel creates a checkers program that improves through self-play",
      category: "Game AI",
      importance: "Innovation",
      keyFigures: ["Arthur Samuel"],
      details: "Samuel's checkers program was one of the first to demonstrate machine learning, improving its play through experience and eventually defeating its creator.",
      impact: "Demonstrated that machines could learn and improve performance through experience.",
      technicalDetails: "Used temporal difference learning and minimax search with learned evaluation functions."
    },
    {
      id: "1956-dartmouth",
      year: "1956",
      title: "Dartmouth Conference",
      description: "The term 'Artificial Intelligence' is coined at the Dartmouth Summer Research Project",
      category: "AI Theory",
      importance: "Foundation",
      keyFigures: ["John McCarthy", "Marvin Minsky", "Nathaniel Rochester", "Claude Shannon", "Allen Newell", "Herbert Simon"],
      details: "The Dartmouth Summer Research Project on Artificial Intelligence was a 6-week workshop that brought together leading researchers to explore machine intelligence. This conference is considered the birth of AI as a field.",
      impact: "Officially established artificial intelligence as a distinct field of study.",
      technicalDetails: "Proposed that 'every aspect of learning or any other feature of intelligence can be so precisely described that a machine can be made to simulate it.'"
    },
    {
      id: "1956-lpt",
      year: "1956",
      title: "Logic Theorist",
      description: "Allen Newell and Herbert Simon create the first AI program",
      category: "AI Programs",
      importance: "Innovation",
      keyFigures: ["Allen Newell", "Herbert Simon", "Cliff Shaw"],
      details: "Logic Theorist (LT) was designed to prove mathematical theorems and was considered the first artificial intelligence program. It successfully proved 38 of the first 52 theorems in Whitehead and Russell's Principia Mathematica.",
      impact: "Demonstrated that machines could perform symbolic reasoning and solve complex logical problems.",
      technicalDetails: "Used heuristic search methods and symbolic manipulation to discover proofs for mathematical theorems."
    },
    {
      id: "1957-perceptron",
      year: "1957",
      title: "Perceptron",
      description: "Frank Rosenblatt develops the perceptron, the first artificial neural network",
      category: "Neural Networks",
      importance: "Breakthrough",
      keyFigures: ["Frank Rosenblatt"],
      details: "The perceptron was the first artificial neural network capable of learning through experience. Rosenblatt demonstrated that it could learn to classify simple patterns, generating significant excitement about the potential of machine learning.",
      impact: "Proved that machines could learn from experience, inspiring decades of neural network research.",
      technicalDetails: "Used a learning rule that adjusted weights based on errors, implementing the first practical learning algorithm for neural networks."
    },
    {
      id: "1958-lisp",
      year: "1958",
      title: "LISP Programming Language",
      description: "John McCarthy creates LISP, which becomes the dominant AI programming language",
      category: "Programming",
      importance: "Foundation",
      keyFigures: ["John McCarthy"],
      details: "LISP (LISt Processing) was designed for symbolic computation and became the preferred language for AI research for decades. Its recursive nature and symbolic manipulation capabilities made it ideal for AI applications.",
      impact: "Provided the primary tool for AI research and development for several decades.",
      technicalDetails: "Featured automatic memory management, dynamic typing, and powerful symbolic processing capabilities."
    },
    {
      id: "1959-machine-learning",
      year: "1959",
      title: "Machine Learning Term Coined",
      description: "Arthur Samuel coins the term 'machine learning' while developing checkers programs",
      category: "AI Theory",
      importance: "Foundation",
      keyFigures: ["Arthur Samuel"],
      details: "Samuel defined machine learning as a 'field of study that gives computers the ability to learn without being explicitly programmed' while working on checkers-playing programs at IBM.",
      impact: "Established machine learning as a distinct approach within artificial intelligence.",
      technicalDetails: "His checkers program could improve its performance through self-play, demonstrating practical machine learning."
    },
    {
      id: "1959-gps",
      year: "1959",
      title: "General Problem Solver",
      description: "Newell and Simon create GPS, an early AI program for general problem solving",
      category: "AI Programs",
      importance: "Innovation",
      keyFigures: ["Allen Newell", "Herbert Simon"],
      details: "The General Problem Solver (GPS) was designed to solve a wide variety of problems using means-ends analysis, representing an early attempt at general artificial intelligence.",
      impact: "Introduced important problem-solving techniques and influenced AI research for decades.",
      technicalDetails: "Used means-ends analysis to reduce differences between current state and goal state."
    },
    {
      id: "1960-adaline",
      year: "1960",
      title: "ADALINE",
      description: "Bernard Widrow develops ADALINE, an adaptive linear neural network",
      category: "Neural Networks",
      importance: "Innovation",
      keyFigures: ["Bernard Widrow", "Ted Hoff"],
      details: "ADALINE (Adaptive Linear Neuron) introduced the least mean squares (LMS) learning algorithm and was one of the first neural networks applied to real-world problems.",
      impact: "Advanced neural network learning algorithms and demonstrated practical applications.",
      technicalDetails: "Used continuous weights and the LMS algorithm for more stable learning than the perceptron."
    },
    {
      id: "1961-unimation",
      year: "1961",
      title: "First Industrial Robot",
      description: "Unimate becomes the first industrial robot used in manufacturing",
      category: "Robotics",
      importance: "Milestone",
      keyFigures: ["George Devol", "Joseph Engelberger"],
      details: "Unimate was installed at a General Motors plant to handle hot metal parts from die-casting machines, marking the beginning of industrial robotics.",
      impact: "Launched the robotics industry and demonstrated practical applications of automated systems.",
      technicalDetails: "Used programmable digital control and hydraulic actuators for precise positioning."
    },
    {
      id: "1962-block-world",
      year: "1962",
      title: "Blocks World",
      description: "Early computer vision and robotics work on manipulating blocks",
      category: "Computer Vision",
      importance: "Innovation",
      keyFigures: ["Larry Roberts"],
      details: "Roberts' thesis work on machine perception of 3D solids laid groundwork for computer vision and robotic manipulation in simplified environments.",
      impact: "Established computer vision as a field and influenced robotics research.",
      technicalDetails: "Used line detection and 3D reconstruction from 2D images to understand geometric shapes."
    },
    {
      id: "1965-eliza",
      year: "1965",
      title: "ELIZA Chatbot",
      description: "Joseph Weizenbaum creates ELIZA, one of the first chatbots",
      category: "NLP",
      importance: "Innovation",
      keyFigures: ["Joseph Weizenbaum"],
      details: "ELIZA was a computer program that could engage in conversations by using pattern matching and substitution methodology. The most famous script, DOCTOR, simulated a Rogerian psychotherapist.",
      impact: "Demonstrated the potential for human-computer interaction through natural language.",
      technicalDetails: "Used simple pattern matching and keyword recognition to generate responses that seemed intelligent."
    },
    {
      id: "1965-dendral",
      year: "1965",
      title: "DENDRAL Expert System",
      description: "Edward Feigenbaum creates DENDRAL for chemical analysis",
      category: "Expert Systems",
      importance: "Innovation",
      keyFigures: ["Edward Feigenbaum", "Bruce Buchanan", "Joshua Lederberg"],
      details: "DENDRAL was one of the first expert systems, designed to identify chemical compounds from mass spectrometer data using knowledge-based reasoning.",
      impact: "Established expert systems as a viable AI approach and showed AI could assist scientific discovery.",
      technicalDetails: "Used rule-based reasoning and domain-specific knowledge to interpret scientific data."
    },
    {
      id: "1966-shakey",
      year: "1966",
      title: "Shakey the Robot",
      description: "SRI develops Shakey, the first mobile robot with reasoning capabilities",
      category: "Robotics",
      importance: "Innovation",
      keyFigures: ["Charles Rosen", "Nils Nilsson", "Bertram Raphael"],
      details: "Shakey combined computer vision, natural language processing, and automated reasoning to navigate and manipulate objects in a simplified environment.",
      impact: "Demonstrated integration of multiple AI technologies in a physical robot system.",
      technicalDetails: "Used STRIPS planning algorithm and A* search for navigation and task planning."
    },
    {
      id: "1967-knn",
      year: "1967",
      title: "k-Nearest Neighbors Algorithm",
      description: "Cover and Hart formalize the k-NN algorithm for pattern classification",
      category: "Machine Learning",
      importance: "Innovation",
      keyFigures: ["Thomas Cover", "Peter Hart"],
      details: "The k-nearest neighbors algorithm became one of the most fundamental and widely-used classification algorithms in machine learning.",
      impact: "Provided a simple yet effective approach to pattern recognition that remains widely used.",
      technicalDetails: "Classifies objects based on the majority class among k nearest neighbors in feature space."
    },
    {
      id: "1969-perceptron-limits",
      year: "1969",
      title: "Perceptron Limitations",
      description: "Minsky and Papert publish 'Perceptrons', highlighting limitations of single-layer networks",
      category: "AI Theory",
      importance: "Setback",
      keyFigures: ["Marvin Minsky", "Seymour Papert"],
      details: "Their book mathematically proved that single-layer perceptrons could not solve non-linearly separable problems like XOR, leading to reduced funding and interest in neural networks for nearly two decades.",
      impact: "Caused the first 'AI Winter' by demonstrating fundamental limitations of existing neural network approaches.",
      technicalDetails: "Showed that perceptrons could only classify linearly separable patterns, severely limiting their applicability."
    },
    {
      id: "1970-prolog",
      year: "1970",
      title: "Prolog Programming Language",
      description: "Alain Colmerauer develops Prolog for logic programming and AI applications",
      category: "Programming",
      importance: "Innovation",
      keyFigures: ["Alain Colmerauer", "Robert Kowalski"],
      details: "Prolog (Programming in Logic) was designed for symbolic reasoning and became important for expert systems and knowledge representation in AI.",
      impact: "Provided a powerful tool for logical reasoning and knowledge-based systems.",
      technicalDetails: "Based on first-order logic and used backward chaining to solve queries through logical inference."
    },
    {
      id: "1972-mycin",
      year: "1972",
      title: "MYCIN Expert System",
      description: "Edward Shortliffe develops MYCIN for medical diagnosis",
      category: "Expert Systems",
      importance: "Innovation",
      keyFigures: ["Edward Shortliffe", "Bruce Buchanan"],
      details: "MYCIN was an expert system for diagnosing bacterial infections and recommending antibiotics, one of the first successful medical AI applications.",
      impact: "Demonstrated AI's potential in medicine and influenced development of medical expert systems.",
      technicalDetails: "Used certainty factors to handle uncertainty in medical knowledge and reasoning."
    },
    {
      id: "1973-assembler",
      year: "1973",
      title: "Computer Vision Progress",
      description: "Significant advances in computer vision and image processing",
      category: "Computer Vision",
      importance: "Innovation",
      keyFigures: ["David Marr", "Shimon Ullman"],
      details: "Marr's computational theory of vision introduced levels of analysis and established computer vision as a rigorous scientific discipline.",
      impact: "Provided theoretical foundation for computer vision that influenced decades of research.",
      technicalDetails: "Proposed computational, algorithmic, and implementational levels of understanding vision."
    },
    {
      id: "1974-ai-winter",
      year: "1974",
      title: "First AI Winter Begins",
      description: "Reduced funding and interest in AI research due to unmet expectations",
      category: "AI Theory",
      importance: "Setback",
      keyFigures: ["James Lighthill"],
      details: "The Lighthill Report in the UK and similar assessments in the US led to significant cuts in AI research funding due to failure to achieve promised breakthroughs.",
      impact: "Forced the AI community to reassess goals and develop more realistic expectations.",
      technicalDetails: "Funding cuts lasted until the early 1980s, slowing progress in neural networks and symbolic AI."
    },
    {
      id: "1975-genetic",
      year: "1975",
      title: "Genetic Algorithms",
      description: "John Holland publishes foundational work on genetic algorithms",
      category: "Evolutionary AI",
      importance: "Innovation",
      keyFigures: ["John Holland"],
      details: "Holland's book 'Adaptation in Natural and Artificial Systems' introduced genetic algorithms as optimization techniques inspired by biological evolution.",
      impact: "Established evolutionary computation as a major AI paradigm for optimization and search.",
      technicalDetails: "Used selection, crossover, and mutation operations to evolve solutions to complex problems."
    },
    {
      id: "1976-am",
      year: "1976",
      title: "AM (Automated Mathematician)",
      description: "Douglas Lenat creates AM, a program that discovers mathematical concepts",
      category: "AI Programs",
      importance: "Innovation",
      keyFigures: ["Douglas Lenat"],
      details: "AM discovered interesting mathematical concepts and conjectures through heuristic search, including rediscovering prime numbers and other mathematical concepts.",
      impact: "Demonstrated AI's potential for creative discovery and automated reasoning in mathematics.",
      technicalDetails: "Used heuristic rules to guide exploration of mathematical concept space."
    },
    {
      id: "1979-bkr",
      year: "1979",
      title: "Stanford Cart",
      description: "Hans Moravec develops autonomous navigation for the Stanford Cart",
      category: "Robotics",
      importance: "Innovation",
      keyFigures: ["Hans Moravec"],
      details: "The Stanford Cart successfully navigated around obstacles using computer vision, taking about 5 hours to cross a chair-filled room.",
      impact: "Advanced autonomous navigation and demonstrated integration of vision and robotics.",
      technicalDetails: "Used stereo vision and world modeling for obstacle avoidance and path planning."
    },
    {
      id: "1980-expert-systems",
      year: "1980",
      title: "Expert Systems Boom",
      description: "XCON and other expert systems demonstrate commercial viability of AI",
      category: "Expert Systems",
      importance: "Breakthrough",
      keyFigures: ["Edward Feigenbaum", "John McDermott"],
      details: "Expert systems like XCON (eXpert CONfigurer) for Digital Equipment Corporation showed that AI could solve real business problems, leading to a resurgence in AI investment.",
      impact: "Proved commercial viability of AI and ended the first AI winter.",
      technicalDetails: "Used rule-based systems to capture human expertise in specific domains."
    },
    {
      id: "1981-japanese",
      year: "1981",
      title: "Japanese Fifth Generation Project",
      description: "Japan launches ambitious AI research program",
      category: "AI Funding",
      importance: "Milestone",
      keyFigures: ["Kazuhiro Fuchi"],
      details: "Japan's Fifth Generation Computer Systems project aimed to create intelligent computers using logic programming and parallel processing.",
      impact: "Spurred international competition in AI research and increased global AI investment.",
      technicalDetails: "Focused on knowledge information processing systems using Prolog and parallel architectures."
    },
    {
      id: "1982-hopfield",
      year: "1982",
      title: "Hopfield Networks",
      description: "John Hopfield introduces recurrent neural networks with associative memory",
      category: "Neural Networks",
      importance: "Innovation",
      keyFigures: ["John Hopfield"],
      details: "Hopfield networks demonstrated how neural networks could store and retrieve patterns, introducing concepts of energy functions and attractor dynamics to neural computation.",
      impact: "Renewed interest in neural networks and introduced important theoretical concepts.",
      technicalDetails: "Used symmetric weights and energy minimization to achieve stable pattern storage and retrieval."
    },
    {
      id: "1983-boltzmann",
      year: "1983",
      title: "Boltzmann Machines",
      description: "Geoffrey Hinton and Terry Sejnowski develop Boltzmann machines",
      category: "Neural Networks",
      importance: "Innovation",
      keyFigures: ["Geoffrey Hinton", "Terry Sejnowski"],
      details: "Boltzmann machines introduced stochastic neural networks that could learn probability distributions over their inputs.",
      impact: "Advanced probabilistic approaches to neural networks and influenced later generative models.",
      technicalDetails: "Used stochastic units and simulated annealing for learning probability distributions."
    },
    {
      id: "1984-cyc",
      year: "1984",
      title: "Cyc Knowledge Base Project",
      description: "Douglas Lenat begins the Cyc project to encode common sense knowledge",
      category: "Knowledge Representation",
      importance: "Innovation",
      keyFigures: ["Douglas Lenat"],
      details: "The Cyc project aimed to create a comprehensive knowledge base of common sense facts and rules that humans take for granted.",
      impact: "Highlighted the importance and difficulty of common sense reasoning in AI.",
      technicalDetails: "Attempted to encode millions of facts and rules about everyday knowledge in logical form."
    },
    {
      id: "1985-nettalk",
      year: "1985",
      title: "NETtalk",
      description: "Terry Sejnowski creates NETtalk, a neural network that learns to pronounce text",
      category: "Neural Networks",
      importance: "Innovation",
      keyFigures: ["Terry Sejnowski", "Charles Rosenberg"],
      details: "NETtalk demonstrated that neural networks could learn complex mappings from text to speech, advancing both neural networks and speech synthesis.",
      impact: "Showed neural networks could learn complex sequential patterns and influenced speech technology.",
      technicalDetails: "Used backpropagation to learn mappings from text characters to phonetic representations."
    },
    {
      id: "1986-backprop",
      year: "1986",
      title: "Backpropagation",
      description: "Rumelhart, Hinton, and Williams popularize backpropagation for training neural networks",
      category: "Neural Networks",
      importance: "Breakthrough",
      keyFigures: ["David Rumelhart", "Geoffrey Hinton", "Ronald Williams"],
      details: "The backpropagation algorithm enabled training of multi-layer neural networks by efficiently computing gradients, solving the credit assignment problem that had limited neural networks.",
      impact: "Made deep neural networks trainable and launched the modern era of neural network research.",
      technicalDetails: "Used the chain rule of calculus to compute gradients layer by layer, enabling efficient training of deep networks."
    },
    {
      id: "1987-ai-winter-2",
      year: "1987",
      title: "Second AI Winter",
      description: "Expert systems market collapses, leading to another period of reduced AI investment",
      category: "AI Theory",
      importance: "Setback",
      keyFigures: [],
      details: "The collapse of the expert systems market and the rise of cheaper desktop computers led to another AI winter, with reduced funding and interest lasting into the 1990s.",
      impact: "Forced another reassessment of AI capabilities and commercial viability.",
      technicalDetails: "Expert systems proved too brittle and expensive to maintain, leading to market collapse."
    },
    {
      id: "1988-elm",
      year: "1988",
      title: "Explanation-Based Learning",
      description: "Development of explanation-based learning techniques",
      category: "Machine Learning",
      importance: "Innovation",
      keyFigures: ["Gerald DeJong", "Raymond Mooney"],
      details: "Explanation-based learning used domain knowledge to generalize from single examples by explaining why examples work.",
      impact: "Advanced machine learning theory and influenced knowledge-based learning approaches.",
      technicalDetails: "Combined deductive reasoning with empirical learning to improve generalization."
    },
    {
      id: "1989-cnn",
      year: "1989",
      title: "Convolutional Neural Networks",
      description: "Yann LeCun develops CNNs for handwritten digit recognition",
      category: "Computer Vision",
      importance: "Innovation",
      keyFigures: ["Yann LeCun"],
      details: "LeCun's convolutional neural networks introduced the concepts of local connectivity, weight sharing, and translation invariance, revolutionizing computer vision.",
      impact: "Established the foundation for modern computer vision and image recognition systems.",
      technicalDetails: "Used convolution operations and pooling to achieve translation invariance and reduce parameters."
    },
    {
      id: "1990-q-learning",
      year: "1990",
      title: "Q-Learning Algorithm",
      description: "Chris Watkins develops Q-learning for reinforcement learning",
      category: "Reinforcement Learning",
      importance: "Innovation",
      keyFigures: ["Chris Watkins"],
      details: "Q-learning provided a model-free approach to reinforcement learning that could learn optimal policies without requiring a model of the environment.",
      impact: "Became one of the most important reinforcement learning algorithms and influenced many later developments.",
      technicalDetails: "Used temporal difference learning to estimate Q-values for state-action pairs."
    },
    {
      id: "1991-td-gammon",
      year: "1991",
      title: "TD-Gammon",
      description: "Gerald Tesauro creates TD-Gammon, a backgammon program using reinforcement learning",
      category: "Game AI",
      importance: "Innovation",
      keyFigures: ["Gerald Tesauro"],
      details: "TD-Gammon used temporal difference learning and neural networks to achieve world-class backgammon play through self-play.",
      impact: "Demonstrated the power of reinforcement learning and self-play for game mastery.",
      technicalDetails: "Combined neural networks with temporal difference learning for position evaluation."
    },
    {
      id: "1993-behavior",
      year: "1993",
      title: "Behavior-Based Robotics",
      description: "Rodney Brooks advances subsumption architecture for robotics",
      category: "Robotics",
      importance: "Innovation",
      keyFigures: ["Rodney Brooks"],
      details: "Brooks' subsumption architecture challenged traditional AI by showing that intelligent behavior could emerge from simple reactive behaviors without central planning.",
      impact: "Revolutionized robotics and influenced embodied AI approaches.",
      technicalDetails: "Used layered behavioral modules that could subsume lower-level behaviors for reactive control."
    },
    {
      id: "1995-svm",
      year: "1995",
      title: "Support Vector Machines",
      description: "Vladimir Vapnik popularizes Support Vector Machines",
      category: "Machine Learning",
      importance: "Innovation",
      keyFigures: ["Vladimir Vapnik", "Corinna Cortes"],
      details: "SVMs provided a principled approach to classification and regression with strong theoretical foundations and excellent practical performance.",
      impact: "Became one of the most successful machine learning algorithms before the deep learning era.",
      technicalDetails: "Used kernel methods and margin maximization for non-linear classification in high-dimensional spaces."
    },
    {
      id: "1995-random-forest",
      year: "1995",
      title: "Random Forest Algorithm",
      description: "Tin Kam Ho introduces random decision forests",
      category: "Machine Learning",
      importance: "Innovation",
      keyFigures: ["Tin Kam Ho", "Leo Breiman"],
      details: "Random forests combined multiple decision trees with randomization to improve accuracy and reduce overfitting.",
      impact: "Became one of the most popular and effective machine learning algorithms.",
      technicalDetails: "Used bagging and feature randomization to create diverse ensembles of decision trees."
    },
    {
      id: "1997-deep-blue",
      year: "1997",
      title: "Deep Blue vs Kasparov",
      description: "IBM's Deep Blue defeats world chess champion Garry Kasparov",
      category: "Game AI",
      importance: "Milestone",
      keyFigures: ["Murray Campbell", "Joe Hoane", "Feng-hsiung Hsu"],
      details: "Deep Blue's victory over the world chess champion demonstrated that computers could outperform humans in complex strategic games, marking a significant milestone in AI capabilities.",
      impact: "Showed the world that AI could exceed human performance in complex cognitive tasks.",
      technicalDetails: "Could evaluate 200 million chess positions per second using specialized chess chips."
    },
    {
      id: "1997-hochreiter",
      year: "1997",
      title: "LSTM Networks",
      description: "Sepp Hochreiter and Jürgen Schmidhuber introduce Long Short-Term Memory networks",
      category: "Neural Networks",
      importance: "Innovation",
      keyFigures: ["Sepp Hochreiter", "Jürgen Schmidhuber"],
      details: "LSTM networks solved the vanishing gradient problem in recurrent neural networks, enabling learning of long-term dependencies in sequential data.",
      impact: "Enabled effective processing of sequential data and became crucial for natural language processing.",
      technicalDetails: "Used gating mechanisms to control information flow and maintain long-term memory."
    },
    {
      id: "1998-pagerank",
      year: "1998",
      title: "PageRank Algorithm",
      description: "Larry Page and Sergey Brin develop PageRank for web search",
      category: "Information Retrieval",
      importance: "Innovation",
      keyFigures: ["Larry Page", "Sergey Brin"],
      details: "PageRank revolutionized web search by ranking pages based on link structure, leading to the founding of Google.",
      impact: "Transformed information retrieval and demonstrated practical applications of graph algorithms.",
      technicalDetails: "Used iterative algorithm to compute importance scores based on link graph structure."
    },
    {
      id: "1999-freebase",
      year: "1999",
      title: "Knowledge Graphs Emergence",
      description: "Early work on structured knowledge representation begins",
      category: "Knowledge Representation",
      importance: "Innovation",
      keyFigures: ["Tim Berners-Lee"],
      details: "The semantic web initiative and early knowledge graph concepts began emerging, leading to structured knowledge representation.",
      impact: "Laid groundwork for modern knowledge graphs used by search engines and AI systems.",
      technicalDetails: "Used RDF and ontologies to represent structured knowledge on the web."
    },
    {
      id: "2001-siri-predecessor",
      year: "2001",
      title: "Early Speech Recognition",
      description: "Advances in speech recognition technology",
      category: "Speech Recognition",
      importance: "Innovation",
      keyFigures: ["BBN Technologies team"],
      details: "Significant improvements in speech recognition accuracy using hidden Markov models and statistical approaches.",
      impact: "Enabled practical speech recognition systems and laid groundwork for voice assistants.",
      technicalDetails: "Used HMMs and statistical language models for continuous speech recognition."
    },
    {
      id: "2003-darpa",
      year: "2003",
      title: "DARPA Grand Challenge",
      description: "DARPA launches autonomous vehicle challenges",
      category: "Robotics",
      importance: "Milestone",
      keyFigures: ["Tony Tether", "Sebastian Thrun"],
      details: "The DARPA Grand Challenge spurred development of autonomous vehicles by offering prizes for self-driving cars.",
      impact: "Accelerated autonomous vehicle research and demonstrated real-world AI applications.",
      technicalDetails: "Required vehicles to navigate desert courses autonomously using sensors and AI."
    },
    {
      id: "2005-stanford-car",
      year: "2005",
      title: "Stanley Wins DARPA Challenge",
      description: "Stanford's Stanley wins the DARPA Grand Challenge",
      category: "Robotics",
      importance: "Milestone",
      keyFigures: ["Sebastian Thrun", "Mike Montemerlo"],
      details: "Stanley successfully completed the 132-mile desert course, proving autonomous vehicles were possible.",
      impact: "Demonstrated feasibility of autonomous vehicles and influenced the automotive industry.",
      technicalDetails: "Used LIDAR, cameras, and machine learning for perception and path planning."
    },
    {
      id: "2006-deep-learning",
      year: "2006",
      title: "Deep Learning Renaissance",
      description: "Geoffrey Hinton coins 'deep learning' and shows deep networks can be trained effectively",
      category: "Deep Learning",
      importance: "Revolution",
      keyFigures: ["Geoffrey Hinton", "Ruslan Salakhutdinov"],
      details: "Hinton's work on deep belief networks and layer-wise pre-training showed that deep neural networks could be trained effectively, launching the deep learning revolution.",
      impact: "Initiated the modern deep learning era and renewed massive interest in neural networks.",
      technicalDetails: "Used unsupervised pre-training followed by supervised fine-tuning to train deep networks."
    },
    {
      id: "2007-fei-fei",
      year: "2007",
      title: "Fei-Fei Li's Vision Work",
      description: "Fei-Fei Li advances computer vision research",
      category: "Computer Vision",
      importance: "Innovation",
      keyFigures: ["Fei-Fei Li"],
      details: "Li's work on object recognition and scene understanding advanced computer vision, leading to the creation of ImageNet.",
      impact: "Advanced computer vision research and influenced the development of large-scale datasets.",
      technicalDetails: "Focused on object recognition, scene understanding, and visual categorization."
    },
    {
      id: "2009-imagenet",
      year: "2009",
      title: "ImageNet Dataset",
      description: "Fei-Fei Li creates ImageNet, a large-scale visual recognition dataset",
      category: "Computer Vision",
      importance: "Foundation",
      keyFigures: ["Fei-Fei Li"],
      details: "ImageNet provided a standardized, large-scale dataset for computer vision research, enabling fair comparison of algorithms and driving progress in visual recognition.",
      impact: "Standardized computer vision research and enabled the development of more sophisticated vision models.",
      technicalDetails: "Contains over 14 million images across 20,000+ categories, hand-annotated for object recognition."
    },
    {
      id: "2010-autoencoder",
      year: "2010",
      title: "Stacked Autoencoders",
      description: "Pascal Vincent and others advance deep autoencoders",
      category: "Deep Learning",
      importance: "Innovation",
      keyFigures: ["Pascal Vincent", "Hugo Larochelle", "Yoshua Bengio"],
      details: "Stacked denoising autoencoders advanced unsupervised learning and pre-training for deep networks.",
      impact: "Improved deep learning training and influenced representation learning.",
      technicalDetails: "Used denoising objectives to learn robust feature representations in deep networks."
    },
    {
      id: "2011-watson",
      year: "2011",
      title: "IBM Watson Wins Jeopardy!",
      description: "IBM's Watson defeats human champions in the quiz show Jeopardy!",
      category: "NLP",
      importance: "Milestone",
      keyFigures: ["David Ferrucci"],
      details: "Watson's victory demonstrated advanced natural language processing and question-answering capabilities, showing AI could understand and respond to complex human language.",
      impact: "Showcased the potential of AI for natural language understanding and knowledge retrieval.",
      technicalDetails: "Used massive parallel processing and advanced NLP techniques to analyze and answer questions."
    },
    {
      id: "2011-alexnet-prep",
      year: "2011",
      title: "GPU Computing for AI",
      description: "Widespread adoption of GPUs for neural network training",
      category: "Computing",
      importance: "Innovation",
      keyFigures: ["Alex Krizhevsky", "NVIDIA team"],
      details: "The use of GPUs dramatically accelerated neural network training, enabling the deep learning revolution.",
      impact: "Made deep learning practical by providing the computational power needed for training large networks.",
      technicalDetails: "Used CUDA and parallel processing to accelerate matrix operations in neural networks."
    },
    {
      id: "2012-alexnet",
      year: "2012",
      title: "AlexNet",
      description: "Krizhevsky's AlexNet wins ImageNet, demonstrating the power of deep CNNs",
      category: "Computer Vision",
      importance: "Breakthrough",
      keyFigures: ["Alex Krizhevsky", "Ilya Sutskever", "Geoffrey Hinton"],
      details: "AlexNet's dramatic victory in the ImageNet competition with a top-5 error rate of 15.3% (vs 26.2% for the runner-up) proved the superiority of deep convolutional neural networks.",
      impact: "Launched the deep learning revolution in computer vision and industry adoption of deep learning.",
      technicalDetails: "Used ReLU activations, dropout regularization, and GPU training to achieve breakthrough performance."
    },
    {
      id: "2013-word2vec",
      year: "2013",
      title: "Word2Vec",
      description: "Tomas Mikolov introduces Word2Vec for learning word embeddings",
      category: "NLP",
      importance: "Innovation",
      keyFigures: ["Tomas Mikolov", "Kai Chen", "Greg Corrado", "Jeffrey Dean"],
      details: "Word2Vec provided efficient methods for learning dense vector representations of words that captured semantic relationships.",
      impact: "Revolutionized natural language processing by providing powerful word representations.",
      technicalDetails: "Used skip-gram and CBOW models to learn word embeddings from large text corpora."
    },
    {
      id: "2013-dqn",
      year: "2013",
      title: "Deep Q-Networks",
      description: "DeepMind develops DQN, combining deep learning with reinforcement learning",
      category: "Reinforcement Learning",
      importance: "Innovation",
      keyFigures: ["Volodymyr Mnih", "David Silver", "Demis Hassabis"],
      details: "DQN successfully learned to play Atari games directly from pixels, demonstrating the potential of deep reinforcement learning.",
      impact: "Launched the field of deep reinforcement learning and influenced many subsequent developments.",
      technicalDetails: "Combined deep neural networks with Q-learning and experience replay for stable learning."
    },
    {
      id: "2014-gans",
      year: "2014",
      title: "GANs Introduced",
      description: "Ian Goodfellow introduces Generative Adversarial Networks",
      category: "Generative AI",
      importance: "Innovation",
      keyFigures: ["Ian Goodfellow"],
      details: "GANs introduced a novel training paradigm where two neural networks compete against each other, enabling the generation of highly realistic synthetic data.",
      impact: "Revolutionized generative modeling and enabled creation of realistic synthetic images, videos, and other data.",
      technicalDetails: "Used adversarial training between a generator and discriminator network to learn data distributions."
    },
    {
      id: "2014-seq2seq",
      year: "2014",
      title: "Sequence-to-Sequence Models",
      description: "Google introduces seq2seq models for machine translation",
      category: "NLP",
      importance: "Innovation",
      keyFigures: ["Ilya Sutskever", "Oriol Vinyals", "Quoc Le"],
      details: "Sequence-to-sequence models using encoder-decoder architectures revolutionized machine translation and other sequence transformation tasks.",
      impact: "Enabled neural machine translation and influenced many other NLP applications.",
      technicalDetails: "Used LSTM encoder-decoder architecture to map input sequences to output sequences of different lengths."
    },
    {
      id: "2014-adam",
      year: "2014",
      title: "Adam Optimizer",
      description: "Diederik Kingma and Jimmy Ba introduce the Adam optimization algorithm",
      category: "Optimization",
      importance: "Innovation",
      keyFigures: ["Diederik Kingma", "Jimmy Ba"],
      details: "Adam became one of the most popular optimization algorithms for training neural networks due to its adaptive learning rates.",
      impact: "Improved neural network training and became the default optimizer for many applications.",
      technicalDetails: "Combined momentum and adaptive learning rates with bias correction for effective optimization."
    },
    {
      id: "2014-dropout",
      year: "2014",
      title: "Dropout Regularization",
      description: "Nitish Srivastava formalizes dropout for preventing overfitting",
      category: "Regularization",
      importance: "Innovation",
      keyFigures: ["Nitish Srivastava", "Geoffrey Hinton"],
      details: "Dropout provided a simple yet effective method for regularizing neural networks by randomly setting neurons to zero during training.",
      impact: "Became a standard technique for preventing overfitting in neural networks.",
      technicalDetails: "Randomly deactivated neurons during training to prevent co-adaptation and improve generalization."
    },
    {
      id: "2015-resnet",
      year: "2015",
      title: "ResNet Architecture",
      description: "Microsoft introduces ResNet with skip connections, enabling very deep networks",
      category: "Computer Vision",
      importance: "Innovation",
      keyFigures: ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
      details: "ResNet's skip connections solved the degradation problem in very deep networks, enabling training of networks with hundreds of layers.",
      impact: "Enabled much deeper neural networks and improved performance across many computer vision tasks.",
      technicalDetails: "Used residual connections to allow gradients to flow directly through the network, preventing vanishing gradients."
    },
    {
      id: "2015-batch-norm",
      year: "2015",
      title: "Batch Normalization",
      description: "Sergey Ioffe and Christian Szegedy introduce batch normalization",
      category: "Optimization",
      importance: "Innovation",
      keyFigures: ["Sergey Ioffe", "Christian Szegedy"],
      details: "Batch normalization normalized layer inputs during training, accelerating training and improving stability.",
      impact: "Became a standard technique in deep learning, enabling faster training and better performance.",
      technicalDetails: "Normalized inputs to each layer using batch statistics and learned scale and shift parameters."
    },
    {
      id: "2016-alphago",
      year: "2016",
      title: "AlphaGo Defeats Lee Sedol",
      description: "DeepMind's AlphaGo defeats world Go champion Lee Sedol",
      category: "Game AI",
      importance: "Milestone",
      keyFigures: ["David Silver", "Demis Hassabis"],
      details: "AlphaGo's victory in Go, a game with more possible positions than atoms in the observable universe, demonstrated the power of combining deep learning with tree search.",
      impact: "Showed AI could master games previously thought impossible for computers and sparked global interest in AI.",
      technicalDetails: "Combined Monte Carlo tree search with deep neural networks trained on human games and self-play."
    },
    {
      id: "2017-transformer",
      year: "2017",
      title: "Transformer Architecture",
      description: "Vaswani et al. introduce the Transformer, revolutionizing NLP",
      category: "NLP",
      importance: "Revolution",
      keyFigures: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan Gomez", "Lukasz Kaiser", "Illia Polosukhin"],
      details: "The Transformer architecture introduced the attention mechanism as the sole building block, eliminating recurrence and convolution while achieving superior performance.",
      impact: "Became the foundation for all modern large language models and revolutionized natural language processing.",
      technicalDetails: "Used self-attention mechanisms and positional encoding to process sequences in parallel, dramatically improving training efficiency."
    },
    {
      id: "2017-alphazero",
      year: "2017",
      title: "AlphaZero",
      description: "DeepMind's AlphaZero masters chess, shogi, and Go from scratch",
      category: "Game AI",
      importance: "Breakthrough",
      keyFigures: ["David Silver", "Julian Schrittwieser"],
      details: "AlphaZero learned to play three different games at superhuman level using only self-play, without human game data.",
      impact: "Demonstrated the power of self-play and general learning algorithms.",
      technicalDetails: "Used Monte Carlo tree search with neural networks trained purely through self-play."
    },
    {
      id: "2018-bert",
      year: "2018",
      title: "BERT",
      description: "Google introduces BERT, achieving state-of-the-art results in NLP tasks",
      category: "NLP",
      importance: "Breakthrough",
      keyFigures: ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
      details: "BERT (Bidirectional Encoder Representations from Transformers) introduced bidirectional training and masked language modeling, achieving breakthrough results across NLP tasks.",
      impact: "Established the pre-training and fine-tuning paradigm that dominates modern NLP.",
      technicalDetails: "Used bidirectional attention and masked language modeling to learn rich contextual representations."
    },
    {
      id: "2018-ulmfit",
      year: "2018",
      title: "ULMFiT Transfer Learning",
      description: "Jeremy Howard and Sebastian Ruder introduce transfer learning for NLP",
      category: "NLP",
      importance: "Innovation",
      keyFigures: ["Jeremy Howard", "Sebastian Ruder"],
      details: "ULMFiT demonstrated effective transfer learning for NLP tasks, showing that language models could be fine-tuned for specific tasks.",
      impact: "Established transfer learning as a key paradigm in NLP before BERT.",
      technicalDetails: "Used pre-trained language models with gradual unfreezing and discriminative learning rates."
    },
    {
      id: "2019-gpt2",
      year: "2019",
      title: "GPT-2",
      description: "OpenAI releases GPT-2, demonstrating impressive text generation capabilities",
      category: "Large Language Models",
      importance: "Innovation",
      keyFigures: ["Alec Radford", "Jeffrey Wu", "Rewon Child", "David Luan"],
      details: "GPT-2's 1.5 billion parameters and impressive text generation led OpenAI to initially withhold the full model due to concerns about misuse.",
      impact: "Demonstrated the potential and risks of large language models, sparking discussions about AI safety.",
      technicalDetails: "Used transformer decoder architecture with 1.5B parameters trained on diverse internet text."
    },
    {
      id: "2019-roberta",
      year: "2019",
      title: "RoBERTa",
      description: "Facebook improves upon BERT with RoBERTa",
      category: "NLP",
      importance: "Innovation",
      keyFigures: ["Yinhan Liu", "Myle Ott", "Naman Goyal"],
      details: "RoBERTa improved BERT's training procedure and achieved better performance on many NLP benchmarks.",
      impact: "Showed the importance of training procedures and hyperparameters in large model performance.",
      technicalDetails: "Removed next sentence prediction and used larger batches and more data for training."
    },
    {
      id: "2019-t5",
      year: "2019",
      title: "T5 (Text-to-Text Transfer Transformer)",
      description: "Google introduces T5, treating every NLP task as text-to-text",
      category: "NLP",
      importance: "Innovation",
      keyFigures: ["Colin Raffel", "Noam Shazeer", "Adam Roberts"],
      details: "T5 unified all NLP tasks under a text-to-text framework, simplifying model architecture and training.",
      impact: "Influenced the design of many subsequent language models and demonstrated the power of unified architectures.",
      technicalDetails: "Used encoder-decoder transformer architecture with text-to-text formulation for all tasks."
    },
    {
      id: "2020-gpt3",
      year: "2020",
      title: "GPT-3",
      description: "OpenAI releases GPT-3, demonstrating unprecedented language generation capabilities",
      category: "Large Language Models",
      importance: "Revolution",
      keyFigures: ["Tom Brown", "Benjamin Mann", "Nick Ryder", "Melanie Subbiah"],
      details: "GPT-3's 175 billion parameters enabled few-shot learning and demonstrated emergent capabilities across diverse tasks without task-specific training.",
      impact: "Showed that scaling language models could lead to emergent capabilities and general intelligence.",
      technicalDetails: "Used 175B parameters and demonstrated in-context learning with just a few examples."
    },
    {
      id: "2020-alphafold1",
      year: "2020",
      title: "AlphaFold First Success",
      description: "DeepMind's AlphaFold shows progress in protein folding prediction",
      category: "Scientific AI",
      importance: "Innovation",
      keyFigures: ["John Jumper", "Richard Evans"],
      details: "AlphaFold demonstrated significant progress in the CASP14 protein folding competition, showing AI's potential for scientific discovery.",
      impact: "Advanced computational biology and demonstrated AI's potential for solving scientific problems.",
      technicalDetails: "Used attention mechanisms and geometric deep learning for protein structure prediction."
    },
    {
      id: "2020-detr",
      year: "2020",
      title: "DETR Object Detection",
      description: "Facebook introduces DETR, applying transformers to object detection",
      category: "Computer Vision",
      importance: "Innovation",
      keyFigures: ["Nicolas Carion", "Francisco Massa"],
      details: "DETR demonstrated that transformers could be effectively applied to computer vision tasks beyond NLP.",
      impact: "Sparked the use of transformers in computer vision and influenced many subsequent vision models.",
      technicalDetails: "Used transformer encoder-decoder architecture with learnable object queries for detection."
    },
    {
      id: "2020-vit",
      year: "2020",
      title: "Vision Transformer (ViT)",
      description: "Google introduces Vision Transformer for image classification",
      category: "Computer Vision",
      importance: "Innovation",
      keyFigures: ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov"],
      details: "ViT showed that pure transformer architectures could achieve excellent performance on image classification without convolutions.",
      impact: "Revolutionized computer vision by demonstrating the effectiveness of transformers for images.",
      technicalDetails: "Treated images as sequences of patches and applied standard transformer architecture."
    },
    {
      id: "2021-alphafold",
      year: "2021",
      title: "AlphaFold Solves Protein Folding",
      description: "DeepMind's AlphaFold accurately predicts protein structures",
      category: "Scientific AI",
      importance: "Breakthrough",
      keyFigures: ["John Jumper", "Richard Evans", "Alexander Pritzel"],
      details: "AlphaFold solved the 50-year-old protein folding problem, accurately predicting 3D protein structures from amino acid sequences.",
      impact: "Revolutionized structural biology and drug discovery, demonstrating AI's potential for scientific breakthroughs.",
      technicalDetails: "Used attention mechanisms and geometric deep learning to predict protein structures with atomic accuracy."
    },
    {
      id: "2021-codex",
      year: "2021",
      title: "OpenAI Codex",
      description: "OpenAI releases Codex, a language model trained on code",
      category: "Code Generation",
      importance: "Innovation",
      keyFigures: ["Mark Chen", "Jerry Tworek", "Heewoo Jun"],
      details: "Codex demonstrated that language models could generate functional code from natural language descriptions.",
      impact: "Launched the era of AI-assisted programming and influenced developer tools.",
      technicalDetails: "Fine-tuned GPT-3 on code repositories to generate Python and other programming languages."
    },
    {
      id: "2021-palm",
      year: "2021",
      title: "PaLM Language Model",
      description: "Google introduces PaLM with 540 billion parameters",
      category: "Large Language Models",
      importance: "Innovation",
      keyFigures: ["Aakanksha Chowdhery", "Sharan Narang"],
      details: "PaLM demonstrated continued scaling benefits in language models with improved performance across many tasks.",
      impact: "Showed that larger language models continued to improve and demonstrated new capabilities.",
      technicalDetails: "Used 540B parameters and demonstrated strong performance on reasoning and multilingual tasks."
    },
    {
      id: "2022-chatgpt",
      year: "2022",
      title: "ChatGPT Launch",
      description: "OpenAI launches ChatGPT, bringing AI to mainstream public attention",
      category: "AI Applications",
      importance: "Cultural Impact",
      keyFigures: ["Sam Altman", "OpenAI Team"],
      details: "ChatGPT reached 100 million users in just 2 months, making AI accessible to the general public and sparking global conversations about AI's impact.",
      impact: "Brought AI into mainstream consciousness and accelerated adoption across industries.",
      technicalDetails: "Based on GPT-3.5 with reinforcement learning from human feedback (RLHF) for improved safety and helpfulness."
    },
    {
      id: "2022-dalle2",
      year: "2022",
      title: "DALL-E 2",
      description: "OpenAI releases DALL-E 2, generating high-quality images from text descriptions",
      category: "Generative AI",
      importance: "Innovation",
      keyFigures: ["Aditya Ramesh", "Prafulla Dhariwal", "Alex Nichol"],
      details: "DALL-E 2 demonstrated unprecedented ability to generate realistic and artistic images from natural language descriptions.",
      impact: "Democratized image creation and sparked new creative applications of AI.",
      technicalDetails: "Used diffusion models and CLIP embeddings to generate high-resolution images from text prompts."
    },
    {
      id: "2022-flamingo",
      year: "2022",
      title: "Flamingo Multimodal Model",
      description: "DeepMind introduces Flamingo for few-shot learning on vision-language tasks",
      category: "Multimodal AI",
      importance: "Innovation",
      keyFigures: ["Jean-Baptiste Alayrac", "Jeff Donahue"],
      details: "Flamingo demonstrated strong few-shot performance on vision-language tasks by bridging pre-trained vision and language models.",
      impact: "Advanced multimodal AI and influenced the development of vision-language models.",
      technicalDetails: "Used perceiver resampler and cross-attention layers to connect vision and language representations."
    },
    {
      id: "2022-minerva",
      year: "2022",
      title: "Minerva Mathematical Reasoning",
      description: "Google introduces Minerva for mathematical reasoning",
      category: "Mathematical AI",
      importance: "Innovation",
      keyFigures: ["Aitor Lewkowycz", "Anders Andreassen"],
      details: "Minerva demonstrated that language models could solve mathematical problems through step-by-step reasoning.",
      impact: "Showed AI's potential for mathematical reasoning and scientific problem-solving.",
      technicalDetails: "Fine-tuned PaLM on mathematical content and used chain-of-thought reasoning for problem solving."
    },
    {
      id: "2023-gpt4",
      year: "2023",
      title: "GPT-4 & Multimodal AI",
      description: "Advanced multimodal AI systems emerge, processing text, images, and more",
      category: "Multimodal AI",
      importance: "Current Era",
      keyFigures: ["OpenAI Team"],
      details: "GPT-4 introduced multimodal capabilities, processing both text and images, while demonstrating improved reasoning and reduced hallucinations.",
      impact: "Marked the beginning of the multimodal AI era with systems that can understand and generate multiple types of content.",
      technicalDetails: "Supports both text and image inputs with improved reasoning capabilities and better alignment with human values."
    },
    {
      id: "2023-llama",
      year: "2023",
      title: "Open Source LLM Revolution",
      description: "Meta releases LLaMA, sparking an open-source large language model movement",
      category: "Large Language Models",
      importance: "Innovation",
      keyFigures: ["Hugo Touvron", "Meta AI Team"],
      details: "LLaMA's release led to a proliferation of open-source language models, democratizing access to large language model technology.",
      impact: "Accelerated research and development by making powerful language models accessible to researchers worldwide.",
      technicalDetails: "Efficient architecture achieving strong performance with fewer parameters than GPT-3."
    },
    {
      id: "2023-claude",
      year: "2023",
      title: "Claude and Constitutional AI",
      description: "Anthropic releases Claude with constitutional AI training",
      category: "AI Safety",
      importance: "Innovation",
      keyFigures: ["Dario Amodei", "Daniela Amodei", "Anthropic Team"],
      details: "Claude demonstrated new approaches to AI safety through constitutional AI, where models are trained to follow a set of principles.",
      impact: "Advanced AI safety research and demonstrated new approaches to aligning AI systems with human values.",
      technicalDetails: "Used constitutional AI training with self-critique and revision to improve harmlessness and helpfulness."
    },
    {
      id: "2023-palm2",
      year: "2023",
      title: "PaLM 2 and Bard",
      description: "Google releases PaLM 2 and integrates it into Bard chatbot",
      category: "Large Language Models",
      importance: "Innovation",
      keyFigures: ["Google Research Team"],
      details: "PaLM 2 improved upon the original PaLM with better multilingual capabilities and reasoning performance.",
      impact: "Demonstrated continued improvements in language models and increased competition in AI assistants.",
      technicalDetails: "Improved training on multilingual and reasoning datasets with better efficiency."
    },
    {
      id: "2023-segment",
      year: "2023",
      title: "Segment Anything Model (SAM)",
      description: "Meta releases SAM for image segmentation",
      category: "Computer Vision",
      importance: "Innovation",
      keyFigures: ["Alexander Kirillov", "Eric Mintun"],
      details: "SAM demonstrated foundation model capabilities for computer vision, enabling segmentation of any object in images.",
      impact: "Advanced computer vision foundation models and democratized image segmentation capabilities.",
      technicalDetails: "Used prompt-based segmentation with a vision transformer backbone trained on diverse segmentation tasks."
    },
    {
      id: "2023-midjourney",
      year: "2023",
      title: "Midjourney and AI Art",
      description: "Midjourney popularizes AI-generated art among mainstream users",
      category: "Generative AI",
      importance: "Cultural Impact",
      keyFigures: ["David Holz", "Midjourney Team"],
      details: "Midjourney made AI art generation accessible to millions of users, democratizing creative tools and sparking debates about AI in art.",
      impact: "Brought AI creativity to the mainstream and influenced artistic practices worldwide.",
      technicalDetails: "Used diffusion models optimized for artistic and aesthetic image generation."
    },
    {
      id: "2024-sora",
      year: "2024",
      title: "Sora Video Generation",
      description: "OpenAI introduces Sora for text-to-video generation",
      category: "Generative AI",
      importance: "Innovation",
      keyFigures: ["OpenAI Team"],
      details: "Sora demonstrated impressive capabilities for generating realistic videos from text descriptions, advancing multimodal AI.",
      impact: "Advanced video generation capabilities and showed potential for AI in film and media production.",
      technicalDetails: "Used diffusion transformers to generate high-quality videos up to one minute long from text prompts."
    },
    {
      id: "2024-claude3",
      year: "2024",
      title: "Claude 3 Family",
      description: "Anthropic releases Claude 3 with improved capabilities",
      category: "Large Language Models",
      importance: "Innovation",
      keyFigures: ["Anthropic Team"],
      details: "Claude 3 demonstrated improvements in reasoning, analysis, and safety compared to previous versions.",
      impact: "Advanced AI safety research and demonstrated continued improvements in language model capabilities.",
      technicalDetails: "Improved training procedures and safety measures with enhanced reasoning and analysis capabilities."
    }
  ]

  const categories = [
    'all', 'Neural Networks', 'AI Theory', 'Computing', 'Programming', 
    'NLP', 'Computer Vision', 'Game AI', 'Generative AI', 'Large Language Models',
    'Expert Systems', 'Deep Learning', 'AI Applications', 'Multimodal AI', 'Scientific AI',
    'Robotics', 'Machine Learning', 'Reinforcement Learning', 'AI Safety', 'Mathematics',
    'Computer Theory', 'Optimization', 'Regularization', 'Speech Recognition', 
    'Information Retrieval', 'Knowledge Representation', 'AI Funding', 'Evolutionary AI',
    'AI Programs', 'Code Generation', 'Mathematical AI', 'Cultural Impact'
  ]

  const decades = ['all', '1840s', '1930s', '1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s', '2020s']

  const filteredEvents = timelineEvents.filter(event => {
    const categoryMatch = selectedCategory === 'all' || event.category === selectedCategory
    const decadeMatch = selectedDecade === 'all' || 
      (selectedDecade === '1840s' && event.year >= '1840' && event.year < '1850') ||
      (selectedDecade === '1930s' && event.year >= '1930' && event.year < '1940') ||
      (selectedDecade === '1940s' && event.year >= '1940' && event.year < '1950') ||
      (selectedDecade === '1950s' && event.year >= '1950' && event.year < '1960') ||
      (selectedDecade === '1960s' && event.year >= '1960' && event.year < '1970') ||
      (selectedDecade === '1970s' && event.year >= '1970' && event.year < '1980') ||
      (selectedDecade === '1980s' && event.year >= '1980' && event.year < '1990') ||
      (selectedDecade === '1990s' && event.year >= '1990' && event.year < '2000') ||
      (selectedDecade === '2000s' && event.year >= '2000' && event.year < '2010') ||
      (selectedDecade === '2010s' && event.year >= '2010' && event.year < '2020') ||
      (selectedDecade === '2020s' && event.year >= '2020')
    
    return categoryMatch && decadeMatch
  })

  const getCategoryColor = (category: string) => {
    const colors = {
      "Neural Networks": "bg-blue-100 text-blue-800 border-blue-200",
      "AI Theory": "bg-purple-100 text-purple-800 border-purple-200",
      "Deep Learning": "bg-green-100 text-green-800 border-green-200",
      "Computer Vision": "bg-yellow-100 text-yellow-800 border-yellow-200",
      "NLP": "bg-red-100 text-red-800 border-red-200",
      "Game AI": "bg-indigo-100 text-indigo-800 border-indigo-200",
      "Generative AI": "bg-pink-100 text-pink-800 border-pink-200",
      "Large Language Models": "bg-orange-100 text-orange-800 border-orange-200",
      "AI Applications": "bg-teal-100 text-teal-800 border-teal-200",
      "Multimodal AI": "bg-cyan-100 text-cyan-800 border-cyan-200",
      "Expert Systems": "bg-amber-100 text-amber-800 border-amber-200",
      "Computing": "bg-gray-100 text-gray-800 border-gray-200",
      "Programming": "bg-slate-100 text-slate-800 border-slate-200",
      "Scientific AI": "bg-emerald-100 text-emerald-800 border-emerald-200",
      "Robotics": "bg-violet-100 text-violet-800 border-violet-200",
      "Machine Learning": "bg-lime-100 text-lime-800 border-lime-200",
      "Reinforcement Learning": "bg-rose-100 text-rose-800 border-rose-200",
      "AI Safety": "bg-sky-100 text-sky-800 border-sky-200",
      "Mathematics": "bg-stone-100 text-stone-800 border-stone-200",
      "Computer Theory": "bg-zinc-100 text-zinc-800 border-zinc-200",
      "Optimization": "bg-neutral-100 text-neutral-800 border-neutral-200",
      "Regularization": "bg-fuchsia-100 text-fuchsia-800 border-fuchsia-200",
      "Speech Recognition": "bg-emerald-100 text-emerald-800 border-emerald-200",
      "Information Retrieval": "bg-blue-100 text-blue-800 border-blue-200",
      "Knowledge Representation": "bg-purple-100 text-purple-800 border-purple-200",
      "AI Funding": "bg-green-100 text-green-800 border-green-200",
      "Evolutionary AI": "bg-yellow-100 text-yellow-800 border-yellow-200",
      "AI Programs": "bg-red-100 text-red-800 border-red-200",
      "Code Generation": "bg-indigo-100 text-indigo-800 border-indigo-200",
      "Mathematical AI": "bg-pink-100 text-pink-800 border-pink-200",
      "Cultural Impact": "bg-orange-100 text-orange-800 border-orange-200"
    }
    return colors[category as keyof typeof colors] || "bg-gray-100 text-gray-800 border-gray-200"
  }

  const getImportanceIcon = (importance: string) => {
    const icons = {
      "Foundation": "🏗️",
      "Breakthrough": "💡",
      "Revolution": "🚀",
      "Milestone": "🏆",
      "Setback": "⚠️",
      "Innovation": "✨",
      "Cultural Impact": "🌍",
      "Current Era": "⭐"
    }
    return icons[importance as keyof typeof icons] || "📍"
  }

  const getImportanceColor = (importance: string) => {
    const colors = {
      "Revolution": "bg-red-100 text-red-800",
      "Breakthrough": "bg-green-100 text-green-800",
      "Foundation": "bg-blue-100 text-blue-800",
      "Innovation": "bg-purple-100 text-purple-800",
      "Milestone": "bg-yellow-100 text-yellow-800",
      "Cultural Impact": "bg-pink-100 text-pink-800",
      "Current Era": "bg-cyan-100 text-cyan-800",
      "Setback": "bg-orange-100 text-orange-800"
    }
    return colors[importance as keyof typeof colors] || "bg-gray-100 text-gray-800"
  }

  return (
    <section className="pt-20 pb-16 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Comprehensive History of <span className="gradient-text">Machine Learning & AI</span>
          </h1>
          <p className="text-lg text-gray-600 max-w-3xl mx-auto">
            Explore the complete journey from mathematical foundations to modern AI systems. 
            Discover the key breakthroughs, influential figures, setbacks, and innovations that shaped artificial intelligence over 180+ years.
          </p>
        </div>

        {/* Filters */}
        <div className="mb-8 space-y-4">
          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-2">Filter by Category:</h3>
            <div className="flex flex-wrap gap-2">
              {categories.map(category => (
                <button
                  key={category}
                  onClick={() => setSelectedCategory(category)}
                  className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                    selectedCategory === category
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {category === 'all' ? 'All Categories' : category}
                </button>
              ))}
            </div>
          </div>
          
          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-2">Filter by Decade:</h3>
            <div className="flex flex-wrap gap-2">
              {decades.map(decade => (
                <button
                  key={decade}
                  onClick={() => setSelectedDecade(decade)}
                  className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                    selectedDecade === decade
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {decade === 'all' ? 'All Decades' : decade}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Timeline */}
        <div className="relative">
          {/* Timeline line */}
          <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-blue-500 via-purple-500 to-pink-500"></div>
          
          <div className="space-y-6">
            {filteredEvents.map((event, index) => (
              <div key={event.id} className="timeline-item">
                <div className="relative">
                  {/* Timeline dot */}
                  <div className="absolute left-6 w-4 h-4 bg-white border-4 border-purple-500 rounded-full z-10"></div>
                  
                  <div className="bg-white rounded-xl shadow-lg p-6 ml-16 card-hover border border-gray-100">
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <span className="text-2xl font-bold text-purple-600">{event.year}</span>
                        <span className="text-2xl">{getImportanceIcon(event.importance)}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`px-3 py-1 rounded-full text-xs font-medium border ${getCategoryColor(event.category)}`}>
                          {event.category}
                        </span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${getImportanceColor(event.importance)}`}>
                          {event.importance}
                        </span>
                      </div>
                    </div>
                    
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">{event.title}</h3>
                    <p className="text-gray-600 mb-4">{event.description}</p>
                    
                    {event.keyFigures.length > 0 && (
                      <div className="mb-4">
                        <span className="text-sm font-medium text-gray-700">Key Figures: </span>
                        <span className="text-sm text-gray-600">{event.keyFigures.join(', ')}</span>
                      </div>
                    )}
                    
                    <button
                      onClick={() => setExpandedEvent(expandedEvent === event.id ? null : event.id)}
                      className="text-purple-600 hover:text-purple-800 text-sm font-medium transition-colors"
                    >
                      {expandedEvent === event.id ? 'Show Less' : 'Learn More'} →
                    </button>
                    
                    {expandedEvent === event.id && (
                      <div className="mt-4 pt-4 border-t border-gray-200 space-y-4">
                        <div>
                          <h4 className="font-semibold text-gray-900 mb-2">Details</h4>
                          <p className="text-gray-600 text-sm">{event.details}</p>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold text-gray-900 mb-2">Impact</h4>
                          <p className="text-gray-600 text-sm">{event.impact}</p>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold text-gray-900 mb-2">Technical Details</h4>
                          <p className="text-gray-600 text-sm">{event.technicalDetails}</p>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Statistics */}
        <div className="mt-16 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Timeline Statistics</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="text-3xl font-bold text-blue-600 mb-2">{timelineEvents.length}</div>
              <div className="text-gray-600">Total Events</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-green-600 mb-2">
                {timelineEvents.filter(e => e.importance === 'Revolution').length}
              </div>
              <div className="text-gray-600">Revolutions</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-purple-600 mb-2">
                {timelineEvents.filter(e => e.importance === 'Breakthrough').length}
              </div>
              <div className="text-gray-600">Breakthroughs</div>
            </div>
            <div className="text-center">
              <div className="text-3xl font-bold text-orange-600 mb-2">180+</div>
              <div className="text-gray-600">Years of Progress</div>
            </div>
          </div>
        </div>

        {/* Key Insights */}
        <div className="mt-12 bg-white rounded-xl shadow-lg p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Key Insights from AI History</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <span className="text-2xl mr-2">🧠</span>
                Neural Network Evolution
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                From McCulloch-Pitts neurons (1943) to modern transformers, neural networks have evolved over 80+ years, 
                with major breakthroughs in the 1980s (backpropagation), 2000s (deep learning), and 2010s (transformers).
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <span className="text-2xl mr-2">📈</span>
                Exponential Acceleration
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                Progress has dramatically accelerated in recent decades, with the 2010s and 2020s seeing unprecedented 
                breakthroughs in language models, computer vision, generative AI, and multimodal systems.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <span className="text-2xl mr-2">🔄</span>
                Cyclical Patterns
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                AI has experienced multiple "winters" (1970s, 1980s) and "springs" driven by breakthroughs and limitations, 
                highlighting the importance of managing expectations and sustained research investment.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <span className="text-2xl mr-2">🌐</span>
                Mainstream Integration
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                Recent years mark AI's transition from research labs to everyday applications, with ChatGPT reaching 
                100M users in 2 months and AI becoming integrated into daily life and business operations.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <span className="text-2xl mr-2">🏗️</span>
                Foundation Building
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                Many foundational concepts (Boolean algebra, neural networks, backpropagation, attention) were developed 
                decades before becoming practical, emphasizing the importance of theoretical research and long-term thinking.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <span className="text-2xl mr-2">⚡</span>
                Hardware Enablement
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                Computing hardware advances (from ENIAC to GPUs) have been crucial enablers of AI progress, 
                with specialized hardware like TPUs and neural chips continuing to drive capabilities forward.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <span className="text-2xl mr-2">📊</span>
                Data-Driven Era
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                The availability of large datasets (ImageNet, internet text, code repositories) has been crucial for 
                training modern AI systems, emphasizing data quality and quantity as key factors in AI development.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <span className="text-2xl mr-2">🤝</span>
                Interdisciplinary Nature
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                AI progress has required contributions from computer science, mathematics, neuroscience, psychology, 
                linguistics, and engineering, demonstrating the value of interdisciplinary collaboration and diverse perspectives.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <span className="text-2xl mr-2">🔬</span>
                Scientific Impact
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                AI has increasingly shown potential for scientific discovery, from protein folding (AlphaFold) to 
                mathematical reasoning (Minerva), suggesting AI could accelerate scientific progress across many fields.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <span className="text-2xl mr-2">🛡️</span>
                Safety Awareness
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                Growing awareness of AI safety, alignment, and ethical considerations has emerged alongside capabilities, 
                with organizations like Anthropic focusing on constitutional AI and responsible development practices.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <span className="text-2xl mr-2">🌍</span>
                Global Competition
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                AI development has become a global competitive priority, with major initiatives from the US, UK, China, 
                EU, and other regions driving investment and research in AI capabilities and infrastructure.
              </p>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
                <span className="text-2xl mr-2">💡</span>
                Open Source Impact
              </h3>
              <p className="text-gray-600 text-sm mb-4">
                The release of models like LLaMA has democratized AI research and development, enabling broader 
                participation and innovation in the AI community while accelerating overall progress.
              </p>
            </div>
          </div>
        </div>

        {/* Influential Figures */}
        <div className="mt-12 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Most Influential Figures</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-white rounded-lg p-4 shadow-sm">
              <h3 className="font-semibold text-gray-900 mb-2">Alan Turing</h3>
              <p className="text-sm text-gray-600">Father of theoretical computer science and AI, created the Turing Test and laid foundations for computation.</p>
            </div>
            <div className="bg-white rounded-lg p-4 shadow-sm">
              <h3 className="font-semibold text-gray-900 mb-2">Geoffrey Hinton</h3>
              <p className="text-sm text-gray-600">Godfather of deep learning, pioneered backpropagation, Boltzmann machines, and launched the deep learning revolution.</p>
            </div>
            <div className="bg-white rounded-lg p-4 shadow-sm">
              <h3 className="font-semibold text-gray-900 mb-2">John McCarthy</h3>
              <p className="text-sm text-gray-600">Coined the term "Artificial Intelligence," created LISP programming language, and organized the Dartmouth Conference.</p>
            </div>
            <div className="bg-white rounded-lg p-4 shadow-sm">
              <h3 className="font-semibold text-gray-900 mb-2">Yann LeCun</h3>
              <p className="text-sm text-gray-600">Pioneer of convolutional neural networks, advanced computer vision, and co-recipient of the Turing Award.</p>
            </div>
            <div className="bg-white rounded-lg p-4 shadow-sm">
              <h3 className="font-semibold text-gray-900 mb-2">Yoshua Bengio</h3>
              <p className="text-sm text-gray-600">Deep learning pioneer, advanced neural language models, and co-recipient of the Turing Award.</p>
            </div>
            <div className="bg-white rounded-lg p-4 shadow-sm">
              <h3 className="font-semibold text-gray-900 mb-2">Fei-Fei Li</h3>
              <p className="text-sm text-gray-600">Created ImageNet dataset, advanced computer vision research, and advocated for AI diversity and ethics.</p>
            </div>
          </div>
        </div>

        {/* Future Outlook */}
        <div className="mt-12 bg-gradient-to-r from-blue-50 to-green-50 rounded-xl p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Looking Forward</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🧠</span>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">Artificial General Intelligence</h3>
              <p className="text-gray-600 text-sm">
                The next major milestone may be achieving human-level general intelligence across all cognitive domains and tasks.
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🌐</span>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">Ubiquitous AI</h3>
              <p className="text-gray-600 text-sm">
                AI will become seamlessly integrated into every aspect of daily life, work, and society.
              </p>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl">🔬</span>
              </div>
              <h3 className="font-semibold text-gray-900 mb-2">Scientific Discovery</h3>
              <p className="text-gray-600 text-sm">
                AI will accelerate scientific breakthroughs in medicine, physics, chemistry, climate science, and beyond.
              </p>
            </div>
          </div>
        </div>
      </div>

      <style jsx>{`
        .gradient-text {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        .timeline-item {
          opacity: 0;
          transform: translateY(20px);
          animation: fadeInUp 0.6s ease-out forwards;
        }
        @keyframes fadeInUp {
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .card-hover {
          transition: all 0.3s ease;
        }
        .card-hover:hover {
          transform: translateY(-2px);
          box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
      `}</style>
    </section>
  )
}