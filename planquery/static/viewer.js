class PlanViewer {
    constructor() {
        this.currentDocument = null;
        this.currentPage = 1;
        this.totalPages = 0;
        this.zoomLevel = 1.0;
        this.regions = [];
        this.selectedRegion = null;
        this.showBboxes = true;
        
        this.initializeElements();
        this.bindEvents();
        this.loadDocuments();
    }
    
    initializeElements() {
        this.documentList = document.getElementById('documentList');
        this.imageContainer = document.getElementById('imageContainer');
        this.pageInfo = document.getElementById('pageInfo');
        this.prevPageBtn = document.getElementById('prevPage');
        this.nextPageBtn = document.getElementById('nextPage');
        this.zoomInBtn = document.getElementById('zoomIn');
        this.zoomOutBtn = document.getElementById('zoomOut');
        this.fitToScreenBtn = document.getElementById('fitToScreen');
        this.zoomLevelSpan = document.getElementById('zoomLevel');
        this.showBboxesToggle = document.getElementById('showBboxes');
        this.regionInfo = document.getElementById('regionInfo');
        this.legend = document.getElementById('legend');
    }
    
    bindEvents() {
        // Page navigation
        this.prevPageBtn.addEventListener('click', () => this.previousPage());
        this.nextPageBtn.addEventListener('click', () => this.nextPage());
        
        // Zoom controls
        this.zoomInBtn.addEventListener('click', () => this.zoomIn());
        this.zoomOutBtn.addEventListener('click', () => this.zoomOut());
        this.fitToScreenBtn.addEventListener('click', () => this.fitToScreen());
        
        // Bbox toggle
        this.showBboxesToggle.addEventListener('change', (e) => {
            this.showBboxes = e.target.checked;
            this.toggleBboxes();
        });
        
        // Hide region info on click outside
        document.addEventListener('click', (e) => {
            if (!this.regionInfo.contains(e.target) && !e.target.classList.contains('bbox')) {
                this.hideRegionInfo();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'ArrowLeft':
                    e.preventDefault();
                    this.previousPage();
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.nextPage();
                    break;
                case '+':
                case '=':
                    e.preventDefault();
                    this.zoomIn();
                    break;
                case '-':
                    e.preventDefault();
                    this.zoomOut();
                    break;
                case '0':
                    e.preventDefault();
                    this.fitToScreen();
                    break;
                case 'b':
                case 'B':
                    e.preventDefault();
                    this.showBboxesToggle.checked = !this.showBboxesToggle.checked;
                    this.showBboxes = this.showBboxesToggle.checked;
                    this.toggleBboxes();
                    break;
            }
        });
    }
    
    async loadDocuments() {
        try {
            const response = await fetch('/api/documents');
            const data = await response.json();
            
            if (data.success) {
                this.renderDocumentList(data.documents);
            } else {
                this.showError('Failed to load documents');
            }
        } catch (error) {
            console.error('Error loading documents:', error);
            this.showError('Error loading documents');
        }
    }
    
    renderDocumentList(documents) {
        if (documents.length === 0) {
            this.documentList.innerHTML = `
                <div class="text-center text-muted">
                    <i class="fas fa-folder-open fa-2x mb-2"></i>
                    <p>No documents found</p>
                    <small>Upload some PDFs to get started</small>
                </div>
            `;
            return;
        }
        
        this.documentList.innerHTML = documents.map(doc => `
            <div class="document-item" data-doc-id="${doc.id}">
                <div class="document-title">${doc.title || doc.file_name}</div>
                <div class="document-meta">
                    <div><i class="fas fa-file-pdf me-1"></i>${doc.file_name}</div>
                    <div><i class="fas fa-layer-group me-1"></i>${doc.total_pages} pages</div>
                    ${doc.discipline ? `<div><i class="fas fa-tag me-1"></i>${doc.discipline}</div>` : ''}
                    <div><i class="fas fa-clock me-1"></i>${new Date(doc.created_at).toLocaleDateString()}</div>
                </div>
            </div>
        `).join('');
        
        // Bind click events
        this.documentList.querySelectorAll('.document-item').forEach(item => {
            item.addEventListener('click', () => {
                const docId = item.dataset.docId;
                this.selectDocument(docId);
            });
        });
    }
    
    async selectDocument(docId) {
        try {
            // Update UI
            this.documentList.querySelectorAll('.document-item').forEach(item => {
                item.classList.toggle('active', item.dataset.docId === docId);
            });
            
            // Load document data
            const response = await fetch(`/api/documents/${docId}`);
            const data = await response.json();
            
            if (data.success) {
                this.currentDocument = data.document;
                this.totalPages = this.currentDocument.total_pages;
                this.currentPage = 1;
                this.loadPage();
            } else {
                this.showError('Failed to load document');
            }
        } catch (error) {
            console.error('Error selecting document:', error);
            this.showError('Error loading document');
        }
    }
    
    async loadPage() {
        if (!this.currentDocument) return;
        
        try {
            // Update page info
            this.updatePageControls();
            
            // Load page image and regions
            const [imageResponse, regionsResponse] = await Promise.all([
                fetch(`/api/documents/${this.currentDocument.id}/pages/${this.currentPage}/image`),
                fetch(`/api/documents/${this.currentDocument.id}/pages/${this.currentPage}/regions`)
            ]);
            
            if (imageResponse.ok && regionsResponse.ok) {
                const imageBlob = await imageResponse.blob();
                const regionsData = await regionsResponse.json();
                
                this.displayImage(imageBlob);
                this.regions = regionsData.regions || [];
                this.renderBboxes();
                this.legend.style.display = 'block';
            } else {
                this.showError('Failed to load page data');
            }
        } catch (error) {
            console.error('Error loading page:', error);
            this.showError('Error loading page');
        }
    }
    
    displayImage(imageBlob) {
        const imageUrl = URL.createObjectURL(imageBlob);
        
        this.imageContainer.innerHTML = `
            <div style="position: relative; display: inline-block;">
                <img id="planImage" class="plan-image" src="${imageUrl}" alt="Plan Image">
            </div>
        `;
        
        const image = document.getElementById('planImage');
        image.onload = () => {
            this.fitToScreen();
            this.renderBboxes();
        };
    }
    
    renderBboxes() {
        if (!this.showBboxes || !this.regions.length) return;
        
        const imageElement = document.getElementById('planImage');
        if (!imageElement) return;
        
        const container = imageElement.parentElement;
        
        // Remove existing bboxes
        container.querySelectorAll('.bbox').forEach(bbox => bbox.remove());
        
        // Add new bboxes
        this.regions.forEach((region, index) => {
            const bbox = document.createElement('div');
            bbox.className = `bbox region-type-${region.region_type || 'other'}`;
            bbox.dataset.regionIndex = index;
            
            // Calculate position relative to image
            const rect = imageElement.getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            
            const scaleX = imageElement.offsetWidth / imageElement.naturalWidth;
            const scaleY = imageElement.offsetHeight / imageElement.naturalHeight;
            
            const x = region.bbox_x1 * scaleX;
            const y = region.bbox_y1 * scaleY;
            const width = (region.bbox_x2 - region.bbox_x1) * scaleX;
            const height = (region.bbox_y2 - region.bbox_y1) * scaleY;
            
            bbox.style.left = `${x}px`;
            bbox.style.top = `${y}px`;
            bbox.style.width = `${width}px`;
            bbox.style.height = `${height}px`;
            
            // Add label
            const label = document.createElement('div');
            label.className = 'bbox-label';
            label.textContent = region.text_type || region.region_type || 'Unknown';
            bbox.appendChild(label);
            
            // Add click handler
            bbox.addEventListener('click', (e) => {
                e.stopPropagation();
                this.selectRegion(index);
            });
            
            container.appendChild(bbox);
        });
    }
    
    selectRegion(index) {
        const region = this.regions[index];
        if (!region) return;
        
        // Update selected region
        this.selectedRegion = index;
        
        // Update bbox styles
        document.querySelectorAll('.bbox').forEach((bbox, i) => {
            bbox.classList.toggle('selected', i === index);
        });
        
        // Show region info
        this.showRegionInfo(region);
    }
    
    showRegionInfo(region) {
        const regionTitle = document.getElementById('regionTitle');
        const regionMeta = document.getElementById('regionMeta');
        const regionText = document.getElementById('regionText');
        
        regionTitle.textContent = `${region.text_type || region.region_type || 'Region'} Details`;
        
        regionMeta.innerHTML = `
            <small class="text-muted">
                Type: ${region.region_type || 'Unknown'} | 
                Confidence: ${(region.confidence * 100).toFixed(1)}% |
                Size: ${Math.round(region.bbox_x2 - region.bbox_x1)} × ${Math.round(region.bbox_y2 - region.bbox_y1)}
            </small>
        `;
        
        regionText.textContent = region.text_content || 'No text content available';
        
        // Position popup near cursor
        const bbox = document.querySelector(`[data-region-index="${this.selectedRegion}"]`);
        if (bbox) {
            const rect = bbox.getBoundingClientRect();
            this.regionInfo.style.left = `${Math.min(rect.right + 10, window.innerWidth - 320)}px`;
            this.regionInfo.style.top = `${Math.max(rect.top, 10)}px`;
        }
        
        this.regionInfo.style.display = 'block';
    }
    
    hideRegionInfo() {
        this.regionInfo.style.display = 'none';
        this.selectedRegion = null;
        document.querySelectorAll('.bbox').forEach(bbox => {
            bbox.classList.remove('selected');
        });
    }
    
    toggleBboxes() {
        const bboxes = document.querySelectorAll('.bbox');
        bboxes.forEach(bbox => {
            bbox.style.display = this.showBboxes ? 'block' : 'none';
        });
        this.legend.style.display = this.showBboxes ? 'block' : 'none';
    }
    
    updatePageControls() {
        this.pageInfo.textContent = `Page ${this.currentPage} of ${this.totalPages}`;
        this.prevPageBtn.disabled = this.currentPage <= 1;
        this.nextPageBtn.disabled = this.currentPage >= this.totalPages;
    }
    
    previousPage() {
        if (this.currentPage > 1) {
            this.currentPage--;
            this.loadPage();
        }
    }
    
    nextPage() {
        if (this.currentPage < this.totalPages) {
            this.currentPage++;
            this.loadPage();
        }
    }
    
    zoomIn() {
        this.zoomLevel = Math.min(this.zoomLevel * 1.2, 5.0);
        this.applyZoom();
    }
    
    zoomOut() {
        this.zoomLevel = Math.max(this.zoomLevel / 1.2, 0.1);
        this.applyZoom();
    }
    
    fitToScreen() {
        const image = document.getElementById('planImage');
        if (!image) return;
        
        const container = this.imageContainer;
        const containerRect = container.getBoundingClientRect();
        
        const scaleX = (containerRect.width - 40) / image.naturalWidth;
        const scaleY = (containerRect.height - 40) / image.naturalHeight;
        
        this.zoomLevel = Math.min(scaleX, scaleY);
        this.applyZoom();
    }
    
    applyZoom() {
        const image = document.getElementById('planImage');
        if (!image) return;
        
        image.style.transform = `scale(${this.zoomLevel})`;
        this.zoomLevelSpan.textContent = `${Math.round(this.zoomLevel * 100)}%`;
        
        // Re-render bboxes after zoom
        setTimeout(() => this.renderBboxes(), 100);
    }
    
    showError(message) {
        this.imageContainer.innerHTML = `
            <div class="text-center text-danger">
                <i class="fas fa-exclamation-triangle fa-3x mb-3"></i>
                <p>${message}</p>
            </div>
        `;
    }
}

// Initialize viewer when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new PlanViewer();
});
