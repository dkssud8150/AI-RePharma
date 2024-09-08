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

        // graphData.nodes에서 유효한 node id 목록을 추출
        const validNodeIds = new Set(graphData.nodes.map(node => node.id));

        // node { data: { id: 'node1', label: 'Node 1' } }
        // relationship { data: { source: 'node1', target: 'node2', label: 'RELATION' } }
        const elements = [
            ...graphData.nodes.map(node => ({
                data: { id: node.id, label: node.name }
            })),
            ...graphData.links
                .filter(link => validNodeIds.has(link.source) && validNodeIds.has(link.target)) // 유효한 노드만 엣지에 추가
                .map(link => ({
                    data: { source: link.source, target: link.target, label: link.relationship }
                }))
        ];

        cytoscape.use(cytoscapeFcose);

        // Cytoscape.js 초기화
        const cy = cytoscape({
            container: document.getElementById('cy'),
            elements: elements,
            // elements: [
            //     ...graphData.nodes.map(node => ({ 
            //         data: { id: node.id, label: node.name } })),
            //     ...graphData.links.map(edge => ({ 
            //         data: { source: edge.source, target: edge.target, label: edge.relationship } }))
            // ],
            style: [
                {
                    selector: 'node',
                    style: {
                        'background-color': '#007bff',
                        'label': 'data(label)',
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
                        'target-arrow-shape': 'triangle',
                        'label': 'data(label)',  // 엣지 레이블 추가
                        'font-size': '10px',
                        'text-rotation': 'autorotate',  // 텍스트가 엣지에 맞춰 회전
                        'color': '#333',  // 텍스트 색상
                        'text-margin-y': -10  // 텍스트 위치 조정
                    }
                }
            ],
            layout: {
                name: 'fcose',
                fit: true,
                padding: 50,
                animate: true,
                randomize: true,
                nodeRepulsion: 10000, // 노드간 반발력
                idealEdgeLength: 150, // 엣지의 이상적인 길이
                edgeElasticity: 0.5, // 엣지의 탄성
                spacingFactor: 1.5 // 전체 노드 간격 조정 비율
            }
        });

        // 그래프를 중앙에 배치
        cy.ready(function() {
            cy.fit();
            cy.center();
        });
    })
    .finally(() => {
        loadingDiv.style.display = 'none';
    });
});
