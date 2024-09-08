document.getElementById('question-form').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const question = document.getElementById('question').value;
    const conversationDiv = document.getElementById('conversation');
    const loadingDiv = document.getElementById('loading');

    loadingDiv.style.display = 'block';

    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: question })
    })
    .then(response => response.json())
    .then(data => {
        const questionBlock = document.createElement('div');
        questionBlock.className = 'question-block';
        questionBlock.innerHTML = `<strong>Q:</strong> ${data.question}`;
        
        const answerBlock = document.createElement('div');
        answerBlock.className = 'answer-block';
        answerBlock.innerHTML = `<strong>A:</strong> ${data.answer}`;
        
        conversationDiv.appendChild(questionBlock);
        conversationDiv.appendChild(answerBlock);
        
        document.getElementById('question').value = '';

        // 시각화할 그래프 데이터
        const graphData = data.graphData;

        // Cytoscape.js 초기화
        const cy = cytoscape({
            container: document.getElementById('cy'),
            elements: graphData.elements,
            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': '#007bff',
                        'label': 'data(id)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'width': '60px',
                        'height': '60px',
                        'font-size': '10px',
                        'color': '#fff',
                        'text-outline-width': 2,
                        'text-outline-color': '#007bff'
                    }
                },
                {
                    selector: 'edge',
                    style: {
                        'width': 3,
                        'line-color': '#ccc',
                        'target-arrow-color': '#ccc',
                        'target-arrow-shape': 'triangle'
                    }
                }
            ],
            layout: {
                name: 'fcose', // 다른 레이아웃을 사용해보세요: 'concentric', 'grid', 'fcose'
                fit: true, // 전체 화면에 맞추기
                padding: 50,
                animate: true,
                randomize: true,
                nodeRepulsion: 4500, // 노드 간의 간격을 조절하여 겹침 방지
                idealEdgeLength: 100, // 엣지의 이상적인 길이
                edgeElasticity: 100, // 엣지의 탄성 조정
                spacingFactor: 1.2 // 그래프의 간격 조정
            }
        });

        // 그래프를 중앙에 배치
        cy.ready(function() {
            cy.fit();
            cy.center();
        });
    })
    .catch(error => console.error('Error:', error))
    .finally(() => {
        loadingDiv.style.display = 'none';
    });
});
