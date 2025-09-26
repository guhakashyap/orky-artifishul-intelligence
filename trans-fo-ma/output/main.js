// ORKY TRANSFO'MA' MAIN JAVASCRIPT
// WAAAGH! Let's make dis website propa Orky!

class OrkyTransformer {
    constructor() {
        this.vocabulary = {
            "WAAAGH": 0,
            "ORK": 1,
            "DAKKA": 2,
            "BOSS": 3,
            "BOYZ": 4,
            "FIGHT": 5,
            "WIN": 6,
            "<PAD>": 7,
            "<START>": 8,
            "<END>": 9
        };
        
        this.components = {
            "OrkAttentionHead": {
                title: "ORK ATTENTION HEAD",
                description: "DIS IS A SINGLE ORK HEAD DAT LOOKS AT WORDS! Each Ork head has three jobs: QUERY ('WOT AM I LOOKIN' FOR?'), KEY ('WOT AM I?'), and VALUE ('WOT DO I KNOW?'). Da Ork looks at all da other words, figures out which ones are most relevant to what he's lookin' for, and den combines all da important information together!"
            },
            "MultiHeadOrkAttention": {
                title: "MULTI-HEAD ORK ATTENTION",
                description: "DIS IS WHERE WE GET A WHOLE MOB OF ORK HEADS WORKIN' TOGETHER! Instead of just one Ork tryin' to figure everything out, we get multiple Orks (heads) to look at da problem from different angles. Den we combine all their answers! It's like havin' a whole mob of Orks all shoutin' their opinions, and den da Boss Ork makes da final decision!"
            },
            "OrkFeedForward": {
                title: "ORK FEED FORWARD",
                description: "DIS IS DA ORK'S BRAIN PROCESSIN' CENTER! After da Ork heads figure out which words are important, dis part takes all dat information and processes it through da Ork's brain. It's like da Ork thinkin' really hard about what he learned and comin' up with a better answer. Da Ork brain expands da information to think about it more deeply, den compresses it back to da right size!"
            },
            "OrkLayerNorm": {
                title: "ORK LAYER NORMALIZATION",
                description: "DIS IS ORK DISCIPLINE! Sometimes Orks get too excited and their numbers get too big or too small. Dis layer keeps 'em in check by normalizin' da values so dey stay in a reasonable range. It's like da Ork Boss keepin' da boyz in line! It calculates da average of all da values, figures out how spread out dey are, and normalizes everything so it's in a nice, controlled range."
            }
        };
        
        this.attentionChart = null;
        this.particleSystem = null;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.createParticleSystem();
        this.populateComponents();
        this.setupScrollReveal();
        this.initializeAttentionChart();
        this.displayVocabulary();
        this.startTypewriterEffect();
    }
    
    setupEventListeners() {
        // Navigation
        document.getElementById('start-exploring').addEventListener('click', () => {
            document.getElementById('transformer-section').scrollIntoView({ behavior: 'smooth' });
        });
        
        document.getElementById('view-code').addEventListener('click', () => {
            document.getElementById('code-section').scrollIntoView({ behavior: 'smooth' });
        });
        
        // Attention analyzer
        document.getElementById('analyze-attention').addEventListener('click', () => {
            this.analyzeAttention();
        });
        
        // Vocabulary builder
        document.getElementById('add-word').addEventListener('click', () => {
            this.addVocabularyWord();
        });
        
        // Code copy
        document.getElementById('copy-code').addEventListener('click', () => {
            this.copyCode();
        });
        
        // Modal controls
        document.getElementById('close-modal').addEventListener('click', () => {
            this.closeModal();
        });
        
        // Enter key support
        document.getElementById('orky-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.analyzeAttention();
            }
        });
        
        document.getElementById('new-word-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                document.getElementById('word-meaning').focus();
            }
        });
        
        document.getElementById('word-meaning').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.addVocabularyWord();
            }
        });
    }
    
    createParticleSystem() {
        // Create particle system using p5.js for Orky atmosphere
        const sketch = (p) => {
            let particles = [];
            
            p.setup = () => {
                const canvas = p.createCanvas(p.windowWidth, p.windowHeight);
                canvas.parent('particle-container');
                canvas.style('position', 'fixed');
                canvas.style('top', '0');
                canvas.style('left', '0');
                canvas.style('z-index', '1');
                canvas.style('pointer-events', 'none');
                
                // Create initial particles
                for (let i = 0; i < 50; i++) {
                    particles.push(new Particle(p));
                }
            };
            
            p.draw = () => {
                p.clear();
                
                // Update and draw particles
                for (let i = particles.length - 1; i >= 0; i--) {
                    particles[i].update();
                    particles[i].display();
                    
                    if (particles[i].isDead()) {
                        particles.splice(i, 1);
                    }
                }
                
                // Add new particles occasionally
                if (p.random() < 0.02) {
                    particles.push(new Particle(p));
                }
            };
            
            p.windowResized = () => {
                p.resizeCanvas(p.windowWidth, p.windowHeight);
            };
            
            class Particle {
                constructor(p) {
                    this.p = p;
                    this.x = p.random(p.width);
                    this.y = p.height + 10;
                    this.vx = p.random(-0.5, 0.5);
                    this.vy = p.random(-2, -0.5);
                    this.life = 255;
                    this.decay = p.random(1, 3);
                    this.size = p.random(2, 6);
                    this.color = p.random(['#DC2626', '#EAB308', '#1E3A8A', '#EA580C']);
                }
                
                update() {
                    this.x += this.vx;
                    this.y += this.vy;
                    this.life -= this.decay;
                }
                
                display() {
                    this.p.push();
                    this.p.translate(this.x, this.y);
                    this.p.noStroke();
                    this.p.fill(this.color + Math.floor(this.life / 255 * 100).toString(16).padStart(2, '0'));
                    this.p.ellipse(0, 0, this.size);
                    this.p.pop();
                }
                
                isDead() {
                    return this.life <= 0 || this.y < -10;
                }
            }
        };
        
        new p5(sketch);
    }
    
    populateComponents() {
        const container = document.getElementById('transformer-components');
        
        Object.entries(this.components).forEach(([key, component]) => {
            const componentDiv = document.createElement('div');
            componentDiv.className = 'transformer-component rounded-lg p-6 text-center';
            componentDiv.innerHTML = `
                <div class="ork-glyph mb-4">⚙️</div>
                <h4 class="ork-title text-xl font-bold mb-2">${component.title}</h4>
                <p class="text-sm text-gray-300">Click to learn more!</p>
            `;
            
            componentDiv.addEventListener('click', () => {
                this.showComponentModal(key, component);
            });
            
            container.appendChild(componentDiv);
        });
    }
    
    showComponentModal(key, component) {
        const modal = document.getElementById('component-modal');
        const title = document.getElementById('modal-title');
        const content = document.getElementById('modal-content');
        
        title.textContent = component.title;
        content.innerHTML = `
            <p class="mb-4">${component.description}</p>
            <div class="bg-gray-800 p-4 rounded-lg">
                <h5 class="font-bold text-yellow-400 mb-2">TECHNICAL DETAILS:</h5>
                <p class="text-sm text-gray-300">This component is part of the Orky Transformer architecture and demonstrates how Ork technology somehow works through the power of belief!</p>
            </div>
        `;
        
        modal.classList.remove('hidden');
        
        // Animate modal appearance
        anime({
            targets: modal.querySelector('.mekboy-panel'),
            scale: [0.8, 1],
            opacity: [0, 1],
            duration: 300,
            easing: 'easeOutBack'
        });
    }
    
    closeModal() {
        const modal = document.getElementById('component-modal');
        
        anime({
            targets: modal.querySelector('.mekboy-panel'),
            scale: [1, 0.8],
            opacity: [1, 0],
            duration: 200,
            easing: 'easeInBack',
            complete: () => {
                modal.classList.add('hidden');
            }
        });
    }
    
    analyzeAttention() {
        const input = document.getElementById('orky-input').value.trim();
        if (!input) {
            this.showNotification('Enter a propa Orky sentence first!', 'warning');
            return;
        }
        
        // Tokenize the input
        const words = input.split(/\s+/);
        const tokens = words.map(word => word.toUpperCase());
        
        // Display tokenized words
        this.displayTokenizedWords(tokens);
        
        // Generate fake attention data for visualization
        const attentionData = this.generateAttentionData(tokens);
        
        // Update the attention chart
        this.updateAttentionChart(tokens, attentionData);
        
        this.showNotification('WAAAGH! Attention analyzed!', 'success');
    }
    
    displayTokenizedWords(tokens) {
        const container = document.getElementById('tokenized-words');
        container.innerHTML = '';
        
        tokens.forEach((token, index) => {
            const tokenDiv = document.createElement('div');
            tokenDiv.className = 'ork-vocab-word';
            tokenDiv.textContent = `${token} (${index})`;
            container.appendChild(tokenDiv);
        });
        
        // Animate tokens
        anime({
            targets: '.ork-vocab-word',
            scale: [0, 1],
            opacity: [0, 1],
            delay: anime.stagger(100),
            duration: 500,
            easing: 'easeOutBack'
        });
    }
    
    generateAttentionData(tokens) {
        const data = [];
        const n = tokens.length;
        
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                // Generate attention weights that favor nearby words and self-attention
                let weight = Math.random() * 0.5 + 0.1;
                if (i === j) weight = Math.random() * 0.3 + 0.7; // Self-attention is stronger
                if (Math.abs(i - j) === 1) weight += 0.2; // Adjacent words get bonus
                
                data.push([i, j, Math.min(weight, 1)]);
            }
        }
        
        return data;
    }
    
    initializeAttentionChart() {
        const chartDom = document.getElementById('attention-chart');
        this.attentionChart = echarts.init(chartDom);
        
        // Initial empty chart
        const option = {
            title: {
                text: 'Click "Analyze Attention" to see Orky magic!',
                textStyle: { color: '#EAB308', fontSize: 16 }
            },
            grid: { show: false }
        };
        
        this.attentionChart.setOption(option);
    }
    
    updateAttentionChart(tokens, attentionData) {
        const option = {
            title: {
                text: 'ORK ATTENTION PATTERNS',
                textStyle: { color: '#EAB308', fontSize: 18, fontWeight: 'bold' }
            },
            tooltip: {
                position: 'top',
                formatter: function(params) {
                    return `From "${tokens[params.data[1]]}" to "${tokens[params.data[0]]}": ${(params.data[2] * 100).toFixed(1)}% attention`;
                }
            },
            grid: {
                height: '70%',
                top: '15%',
                left: '10%',
                right: '10%'
            },
            xAxis: {
                type: 'category',
                data: tokens,
                splitArea: { show: true },
                axisLabel: { color: '#F3F4F6', rotate: 45 }
            },
            yAxis: {
                type: 'category',
                data: tokens,
                splitArea: { show: true },
                axisLabel: { color: '#F3F4F6' }
            },
            visualMap: {
                min: 0,
                max: 1,
                calculable: true,
                orient: 'horizontal',
                left: 'center',
                bottom: '5%',
                inRange: {
                    color: ['#1E3A8A', '#DC2626', '#EAB308']
                },
                textStyle: { color: '#F3F4F6' }
            },
            series: [{
                name: 'Attention Weight',
                type: 'heatmap',
                data: attentionData,
                label: {
                    show: false
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(220, 38, 38, 0.5)'
                    }
                }
            }]
        };
        
        this.attentionChart.setOption(option, true);
        
        // Animate chart appearance
        anime({
            targets: '#attention-chart',
            scale: [0.9, 1],
            opacity: [0.5, 1],
            duration: 800,
            easing: 'easeOutQuart'
        });
    }
    
    addVocabularyWord() {
        const wordInput = document.getElementById('new-word-input');
        const meaningInput = document.getElementById('word-meaning');
        
        const word = wordInput.value.trim().toUpperCase();
        const meaning = meaningInput.value.trim();
        
        if (!word || !meaning) {
            this.showNotification('Enter both word and meaning, ya git!', 'warning');
            return;
        }
        
        if (this.vocabulary[word] !== undefined) {
            this.showNotification('Dat word already exists in da vocabulary!', 'warning');
            return;
        }
        
        // Add to vocabulary (simulate with object)
        const newId = Object.keys(this.vocabulary).length;
        this.vocabulary[word] = newId;
        
        // Clear inputs
        wordInput.value = '';
        meaningInput.value = '';
        
        // Update display
        this.displayVocabulary();
        
        this.showNotification(`"${word}" added to da Orky vocabulary!`, 'success');
    }
    
    displayVocabulary() {
        const container = document.getElementById('vocabulary-display');
        container.innerHTML = '';
        
        Object.entries(this.vocabulary).forEach(([word, id]) => {
            const wordDiv = document.createElement('div');
            wordDiv.className = 'bg-gray-800 p-3 rounded-lg mb-2 flex justify-between items-center';
            wordDiv.innerHTML = `
                <span class="font-bold text-yellow-400">${word}</span>
                <span class="text-gray-400 text-sm">ID: ${id}</span>
            `;
            container.appendChild(wordDiv);
        });
    }
    
    copyCode() {
        const codeText = `class OrkAttentionHead(nn.Module):
    """
    DIS IS A SINGLE ORK HEAD DAT LOOKS AT WORDS!
    Each Ork head has three jobs:
    1. QUERY: "WOT AM I LOOKIN' FOR?"
    2. KEY: "WOT AM I?"
    3. VALUE: "WOT DO I KNOW?"
    """
    
    def __init__(self, da_orky_model_size, da_orky_head_size):
        super().__init__()
        self.da_orky_head_size = da_orky_head_size
        
        # Dese 'ere are da Ork's three brain bitz
        self.lookin_fer = nn.Linear(da_orky_model_size, da_orky_head_size)
        self.wot_am_i = nn.Linear(da_orky_model_size, da_orky_head_size)  
        self.wot_i_know = nn.Linear(da_orky_model_size, da_orky_head_size)`;
        
        navigator.clipboard.writeText(codeText).then(() => {
            this.showNotification('Kode copied to clipboard! WAAAGH!', 'success');
        }).catch(() => {
            this.showNotification('Failed to copy kode. Try again, ya git!', 'error');
        });
    }
    
    setupScrollReveal() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                }
            });
        }, observerOptions);
        
        document.querySelectorAll('.ork-scroll-reveal').forEach(el => {
            observer.observe(el);
        });
    }
    
    startTypewriterEffect() {
        const typewriterElement = document.querySelector('.typewriter');
        const texts = [
            'Da Orky Transfo\'ma\' works by havin\' lots of Ork heads lookin\' at words and figurin\' out which words is most important to each other...',
            'It\'s like havin\' a whole mob of Orks shoutin\' at each other about what dey fink is important!',
            'NOW WIF MORE ORKY VARIABLES AND COMMENTS SO EVEN DA DUMBEST GROT CAN UNDERSTAND!',
            'RED WUNZ GO FASTA! BLUE IS FOR LUCK! YELLOW MAKES BIGGER BOOMS!'
        ];
        
        let currentText = 0;
        
        setInterval(() => {
            currentText = (currentText + 1) % texts.length;
            
            anime({
                targets: typewriterElement,
                opacity: [1, 0],
                duration: 500,
                complete: () => {
                    typewriterElement.textContent = texts[currentText];
                    anime({
                        targets: typewriterElement,
                        opacity: [0, 1],
                        duration: 500
                    });
                }
            });
        }, 5000);
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `fixed top-24 right-6 z-50 p-4 rounded-lg shadow-lg max-w-sm ${
            type === 'success' ? 'bg-green-600' :
            type === 'warning' ? 'bg-yellow-600' :
            type === 'error' ? 'bg-red-600' : 'bg-blue-600'
        }`;
        notification.innerHTML = `
            <div class="flex items-center space-x-2">
                <span class="text-xl">${
                    type === 'success' ? '✅' :
                    type === 'warning' ? '⚠️' :
                    type === 'error' ? '❌' : 'ℹ️'
                }</span>
                <span class="font-bold">${message}</span>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        anime({
            targets: notification,
            translateX: [300, 0],
            opacity: [0, 1],
            duration: 300,
            easing: 'easeOutBack'
        });
        
        // Remove after 3 seconds
        setTimeout(() => {
            anime({
                targets: notification,
                translateX: [0, 300],
                opacity: [1, 0],
                duration: 300,
                easing: 'easeInBack',
                complete: () => {
                    notification.remove();
                }
            });
        }, 3000);
    }
}

// Initialize the Orky Transformer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new OrkyTransformer();
    
    // Add scroll reveal to elements
    document.querySelectorAll('section').forEach(section => {
        section.classList.add('ork-scroll-reveal');
    });
});