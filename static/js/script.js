document.addEventListener('DOMContentLoaded', function(){
    const fileInput = document.getElementById('file-upload');  // 파일 입력 요소
    const questionInput = document.getElementById('question');
    const submitButton = document.querySelector('.submit-button');
    const fileNameSpan = document.getElementById('file-name');  // 파일 이름을 표시할 요소
    
    // 파일이 선택된 경우 파일 이름 표시
    fileInput.addEventListener('change', function(event) {
        if (fileInput.files.length > 0) {
            const fileName = fileInput.files[0].name;  // 파일 이름 추출
            fileNameSpan.textContent = fileName;  // 파일 이름을 화면에 표시
        } else {
            fileNameSpan.textContent = '';  // 파일이 없으면 빈 텍스트로 초기화
        }
        checkFromValidity();
    });

    questionInput.addEventListener('input', function() {
        checkFromValidity();
    });

    function checkFromValidity() {
        if (questionInput.value != '' || fileInput.files.length > 0) {
            submitButton.disabled = false;
        } else {
            submitButton.disabled = true;
        }
    };

    document.getElementById('question-form').addEventListener('submit', function(event) {
        event.preventDefault();

        const question = questionInput.value;
        const loadingDiv = document.getElementById('loading');

        loadingDiv.style.display = 'block';

        // FormData 객체 생성
        const formData = new FormData();
        formData.append('question', question); // 질문 추가

        // 파일이 선택되었을 때만 FormData에 파일 추가
        if (fileInput.files.length > 0) {
            formData.append('file', fileInput.files[0]); // 파일 추가
        }

        fetch('/ask', {
            method: 'POST',
            body: formData, // FormData 전송
        })
        .then(response => response.json())
        .then(data => {
            // 시각화할 그래프 데이터
            const graphData = data.graphData;

            // graphData.nodes에서 유효한 node id 목록을 추출
            const validNodeIds = new Set(graphData.nodes.map(node => String(node.id)));
            const filteredLinks = graphData.links.filter(link => validNodeIds.has(String(link.source)) && validNodeIds.has(String(link.target)));

            // node { data: { id: 'node1', label: 'Node 1' } }
            // relationship { data: { source: 'node1', target: 'node2', label: 'RELATION' } }
            const elements = [
                ...graphData.nodes
                .filter(node => {
                    const hasConnectedEdges = graphData.links.some(link => 
                        link.source === node.id || link.target === node.id
                    );
                    return hasConnectedEdges;
                })
                .map(node => ({
                    data: { id: String(node.id), label: node.name }
                })),
                ...filteredLinks.map(link => ({
                    data: { source: String(link.source), target: String(link.target), label: link.relationship }
                }))
            ];

            cytoscape.use(cytoscapeFcose);

            // Cytoscape.js 초기화
            const cy = cytoscape({
                container: document.getElementById('cy'),
                elements: elements,
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
                            'text-rotation': 'autorotate',
                            'color': '#333',
                            'text-margin-y': -10
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
                    spacingFactor: 1.5, // 전체 노드 간격 조정 비율
                }
            });

            // 그래프를 중앙에 배치
            cy.ready(function() {
                cy.nodes().forEach(node => {
                    if (node.connectedEdges().length === 0) {
                        const randomX = Math.random() * 1000 - 500;  // x축 랜덤 값 (-500 ~ 500)
                        const randomY = Math.random() * 1000 - 500;  // y축 랜덤 값 (-500 ~ 500)
                        node.position({ x: randomX, y: randomY });
                    }
                });

                cy.fit();
                cy.center();
            });
        })
        .finally(() => {
            loadingDiv.style.display = 'none';

            // 제출 후 폼 초기화
            questionInput.value = '';
            fileInput.value = ''; // 파일 입력 초기화
            fileNameSpan.textContent = ''; // 파일 이름 초기화
            submitButton.disabled = true; // 다시 비활성화        
        });
    });

});