# **Orky Artifishul Intelligence**

Dis is where we build da smartest gubbinz in da whole galaxy\! We're takin' all dat fancy "Artifishul Intelligence" stuff da Humiez use and makin' it propa' Orky.

### **FOR DA BOYZ\!**

WAAAGH\! Listen up, ya gits\!

You ever wanted a Grot dat knows which Shoota has da most Dakka without even lookin'? Or a Squig dat can sniff out a propa' fight from a mile away? Well, you've come to da right workshop\!

Dis 'ere place is for all you Mekboyz who wanna learn how to make fings *fink*. We write da kode in a way even a Grot can understand, with lots of comments explainin' all da **Kunnin'** and **Killy bitz**. We started with a **Transfo'ma'**, which is like a big mob of Orks all shoutin' at words to figure out what they mean. Now we also got a **Morky Mamba**, which is like a really smart Ork who can remember important stuff and forget da boring stuff! And we got da **Gargant Titans**, which is like a massive Ork war machine with collective WAAAGH memory dat grows stronger over time! Soon, we'll have all sorts of smart gubbinz\!

So grab yer tools, grab yer thinkin' cap (if it ain't too small), and let's get buildin'\! If you've got yer own Orky kode, bring it on\! Da more Kunnin' we build, da bigger our WAAAGH\! gets\!

### **For the Humans**

Welcome\!

This repository is an educational project with a simple goal: to make complex Artificial Intelligence and Machine Learning concepts more accessible and entertaining to learn.

The approach is to implement various ML models and algorithms in Python, but with all variables, functions, and comments written from the perspective of a Warhammer 40,000 Ork. By using analogy, humor, and a simplified (if chaotic) narrative, the code aims to provide a more intuitive understanding of how these systems work.

The project currently includes:

* **Orky Transfo'ma' (trans-fo-ma/):** A from-scratch implementation of the Transformer architecture, the foundation for many modern LLMs. The extensive "Orky" comments explain the purpose of everything from embeddings to multi-head attention.

### **🔵 How da Orky Transfo'ma' Works (Step-by-Step for Gits!)**

**RIGHT DEN, LISTEN UP, YA GIT!** So you wanna know how da "Orky Transfo'ma'" works, eh? Good! A proppa Ork should know how his shiny bitz work before he takes 'em to a WAAAGH!

Forget all dat humie gobbledegook. We'z gonna build this thing step-by-step, like a Mek buildin' a new Killa Kan.

Let's say we wanna teach da machine to answer questions. We yell at it: **"WOTZ DA PLAN?"** and we want it to yell back: **"KRUMP DA GITZ!"**

Here's how da Transfo'ma' does it.

#### **Step 1: Turnin' Yer Yellin' Into Proppa Orky Kunnin' (That's Numbers!)**

A machine is a git. It don't know words. It only knows numbers. So first, we gotta turn our words into numbers.

* **Da Choppa (Tokenizer):** First, we chop up our sentence.
    * "WOTZ", "DA", "PLAN", "?" becomes four bits. We give each bit a number from our big book of Orky words.
    * `WOTZ` -> `5`
    * `DA` -> `2`
    * `PLAN` -> `8`
    * `?` -> `99`
    * So now we got: `[5, 2, 8, 99]`

* **Givin' it Sum Hefta (Embedding):** A number ain't enough. We need to tell da machine *wot kinda word* it is. Is it a stompy word? A shooty word? We turn each number into a list of numbers, like a stat sheet for a Nob.
    * `5` ("WOTZ") might become `[0.1, 0.8, 0.3, ...]`
    * This list of numbers tells da machine wot "WOTZ" means compared to other words.

* **Knowin' Where Ya Stand (Positional Encoding):** The order of words matters! "ORK KRUMPS GIT" is good. "GIT KRUMPS ORK" is bad! So, we add *another* list of numbers that just tells da machine where each word is in the line. Is it first? Second? Third?

Now our words are a bunch of numbers full of meanin' and ready to go into da big Thinkin' Box!

#### **Step 2: Da Thinkin' Box (Da Encoder)**

The Encoder's job is to look at all yer words at once and figure out what you *really* mean. It does this with a special trick called **Self-Attention**, or as I call it, **"Da Big Look-See."**

Imagine a Warboss lookin' at his Boyz before a fight. When he looks at his Nob, he's also lookin' at the Boy with the Big Shoota, and the Grot with the ammo, 'cause they're all important to what the Nob is gonna do.

That's what Da Big Look-See does. For every word, it looks at all the *other* words in the sentence and decides how important they are to each other.

* **Example:** In **"WOTZ DA PLAN?"**, when the machine looks at "PLAN", the Look-See helps it pay more attention to "WOTZ". It learns that "WOTZ" is askin' a question *about* the "PLAN". It connects 'em!

The Encoder does this over and over, getting a better idea of the sentence's meaning each time. When it's done, it spits out a new set of number lists, full of kunnin' and context.

#### **Step 3: Da Yappin' Box (Da Decoder)**

Now we got the meaning of **"WOTZ DA PLAN?"**. The Decoder's job is to start talkin' back. It's gonna build the answer, one word at a time.

1.  **It starts with a "start" token.** Just a little nudge to get it goin'.
2. It uses its *own* Big Look-See, but it's CHEATIN' PROOF. It can only look at the words it has already said. So, when it's thinkin' of the first word, it can't see the second.
3.  **NOW, DA KUNNIN' BIT:** The Decoder takes what it's got so far (just the "start" token) and looks at the super-smart numbers that came out of the Encoder. It asks, "Based on the meanin' of 'WOTZ DA PLAN?', what's the most Orky word I should say first?"

#### **Step 4: Pickin' Da Loudest Word (Da Final Bit)**

The Yappin' Box thinks real hard and spits out a list of probabilities for every single word it knows.

* `KRUMP`: 90% chance (Sounds loud and likely!)
* `DAKKA`: 5% chance (Maybe, but not the best first word)
* `GIT`: 2% chance
* `SQUIG`: 1% chance
* *...and so on for all the other words.*

The machine picks the loudest, most likely word: **"KRUMP"**.

**NOW WE REPEAT!**

The Decoder takes "KRUMP" and feeds it back into itself. Now it's thinkin': "Okay, I've said 'KRUMP'. Based on that, AND the original question 'WOTZ DA PLAN?', what's the next word?"

It runs the numbers again...

* `DA`: 85% chance
* `MORE`: 10% chance
* `A`: 3% chance

It picks **"DA"**. The answer is now "KRUMP DA".

It does it one more time... It's thinkin' "Okay, I've said 'KRUMP DA'... wot's next?" It looks at the meaning of "WOTZ DA PLAN?" and picks the next best word... **"GITZ!"**

Finally, it predicts a special "end of sentence" word, and the machine shuts its gob.

And there ya have it, ya git!

**Yell In:** "WOTZ DA PLAN?"
**Machine Thinks & Yells Back:** "KRUMP DA GITZ!"

That's how an Orky Transfo'ma' works. It ain't magic, it's just proppa, brutal kunnin'! Now get back to work! WAAAGH!

* **Morky Mamba (mam-ba/):** A from-scratch implementation of the Mamba SSM (State Space Model) architecture, featuring selective memory and sequential processing. Unlike Transformers that process all words at once, Mamba processes sequences one word at a time and can selectively remember important information while forgetting unimportant details.

* **Gargant Titans (gargant/):** A from-scratch implementation of the Titans architecture, featuring advanced memory mechanisms and surprise detection. The Gargant is like a massive Ork war machine with collective WAAAGH memory that grows stronger over time, surprise detection that remembers unexpected events, and dynamic memory that learns during battle (test time).

* **Hydra Waaagh (hydra/):** A from-scratch implementation of the Hydra hybrid architecture, combining State Space Models, multi-head attention, and Mixture of Experts. The Hydra is like having multiple Ork Warbosses all specializing in different things but working together - each head focuses on different aspects (local fights, long-range battles, memory, prediction) while sharing the same WAAAGH energy!

* **Hyena Hierarchy (hyena/):** A from-scratch implementation of the Hyena convolutional sequence model, using global convolutions for efficient long-range dependencies. The Hyena is like the WAAAGH energy field connecting all Orks instantly across the battlefield - processing sequences in linear time with global awareness, perfect for handling massive documents and long conversations!

* **MoE Waaagh (moe/):** A from-scratch implementation of the Mixture of Experts architecture, featuring specialized expert networks with learned routing and sparse activation. The MoE is like having different Ork clans specializing in different jobs - each clan is really good at their specialty, and the smart Warboss (router) assigns tasks to the right clans!

* **Krork-HRM (krork-hrm/):** A from-scratch implementation of the Hierarchical Reasoning Model (HRM) architecture, featuring brain-inspired hierarchical processing with high-level Warboss planning and low-level Boyz execution. The Krork-HRM is like the ancient Krork intelligence from before the fall - super-intelligent, hierarchical, and capable of complex reasoning with minimal training data!

### **Key Differences: Orky vs Humie Comparison Table**

| **Orky Name** | **Architecture** | **Orky Description** | **Humie Description** | **Best For** | **Complexity** |
|---------------|------------------|---------------------|----------------------|--------------|----------------|
| **🔵 Orky Transfo'ma'** | Transformer | Like a big mob of Orks all shoutin' at words at da same time! All words can "talk" to each other simultaneously! | Parallel processing with attention mechanisms. All tokens can attend to each other simultaneously. | Understanding relationships between words, general language tasks | O(n²) - Quadratic |
| **🔴 Morky Mamba** | Mamba SSM | Like a really smart Ork who reads words one by one and remembers selectively! Can forget boring stuff! | Sequential processing with selective memory. Processes tokens one at a time with selective state updates. | Long sequences, memory management, efficient processing | O(n) - Linear |
| **🟣 Gargant Titans** | Titans | Like a massive Ork war machine with collective WAAAGH memory! Surprise detection remembers unexpected events! | Advanced memory mechanisms with surprise detection. Collective memory that grows stronger over time. | Complex memory tasks, surprise detection, dynamic learning | O(n) - Linear |
| **🟢 Hydra Waaagh** | Hydra Hybrid | Like multiple Ork Warbosses all specializin' in different fings but workin' together! Each head has a specialty! | Hybrid architecture combining SSMs, multi-head attention, and MoE. Each head specializes in different aspects. | Complex tasks needing multiple capabilities, hybrid processing | O(n) - Linear |
| **🟡 Hyena Hierarchy** | Hyena Convolutional | Like da WAAAGH energy field connectin' all Orks instantly across da battlefield! No attention matrices needed! | Global convolutions for efficient long-range dependencies. Linear complexity with global awareness. | Massive documents, long conversations, global dependencies | O(n) - Linear |
| **🟠 MoE Waaagh** | Mixture of Experts | Like havin' different Ork clans specializin' in different jobs! Smart Warboss assigns tasks to da right clans! | Mixture of Experts with specialized networks and learned routing. Sparse activation for efficiency. | Complex tasks needing specialized expertise, scalable processing | O(n) - Linear |
| **🧠 Krork-HRM** | Hierarchical Reasoning Model | Like da ancient Krork intelligence from before da fall! Warboss plans strategy, Boyz handle tactics! | Hierarchical Reasoning Model with brain-inspired processing. High-level planning + low-level execution. | Complex reasoning, minimal data, fast inference | O(n) - Linear |

### **🎯 Quick Architecture Guide:**

**🔵 For General Tasks:** Orky Transfo'ma' (parallel processing, attention)
**🔴 For Long Sequences:** Morky Mamba (selective memory, efficiency)
**🟣 For Memory Tasks:** Gargant Titans (advanced memory, surprise detection)
**🟢 For Hybrid Tasks:** Hydra Waaagh (best of all worlds)
**🟡 For Global Dependencies:** Hyena Hierarchy (linear complexity, global awareness)
**🟠 For Specialized Tasks:** MoE Waaagh (expert networks, sparse activation)
**🧠 For Complex Reasoning:** Krork-HRM (hierarchical processing, minimal data)

### **Quick Start**

To see da Orks in action, just run:

```bash
# For da Orky Transfo'ma'
cd trans-fo-ma
python trans-fo-ma1.3.py

# For da Morky Mamba  
cd mam-ba
python mam-ba1.0.py

# For da Gargant Titans
cd gargant
python gargant_titans.py

# For da Hydra Waaagh
cd hydra
python hydra_waaagh.py

# For da Hyena Hierarchy
cd hyena
python hyena_demo.py

# For da MoE Waaagh
cd moe
python moe_demo.py

# For da Krork-HRM
cd krork-hrm
python krork_demo.py
```

All implementations include detailed demonstrations that show how da Orks process information and make predictions!

The plan is to expand this collection to cover other foundational AI/ML topics. Contributions are welcome\! If you have an idea for an "Orky" implementation of an algorithm or model, feel free to open an issue or submit a pull request.

