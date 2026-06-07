function togglePhase(id) {
    const el = document.getElementById(id);
    if (el.classList.contains('hidden')) {
        el.classList.remove('hidden');
    } else {
        el.classList.add('hidden');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-pipeline-btn');
    const statusMsg = document.getElementById('status-message');
    const resultsContainer = document.getElementById('results-container');

    runBtn.addEventListener('click', async () => {
        runBtn.disabled = true;
        statusMsg.classList.remove('hidden', 'success', 'error');
        statusMsg.classList.add('running');
        statusMsg.innerText = 'Running pipeline... This may take a minute depending on the models.';
        resultsContainer.classList.add('hidden');

        try {
            const response = await fetch('/api/run-pipeline', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            const result = await response.json();

            if (result.status === 'success') {
                statusMsg.classList.remove('running');
                statusMsg.classList.add('success');
                statusMsg.innerText = 'Pipeline execution completed successfully!';
                
                populateResults(result.data);
                resultsContainer.classList.remove('hidden');
            } else {
                throw new Error(result.message);
            }
        } catch (error) {
            statusMsg.classList.remove('running');
            statusMsg.classList.add('error');
            statusMsg.innerText = 'Error running pipeline: ' + error.message;
            console.error(error);
        } finally {
            runBtn.disabled = false;
        }
    });
});

function populateResults(data) {
    // Phase 2
    if(data.phase2) {
        document.getElementById('phase2-content').innerHTML = `
            <p><strong>Dataset Shape:</strong> ${data.phase2.shape[0]} rows, ${data.phase2.shape[1]} columns</p>
            <p><strong>Columns:</strong> ${data.phase2.columns.join(', ')}</p>
        `;
    }

    // Phase 3
    if(data.phase3) {
        let html = '<h4>Missing Values</h4><ul>';
        for(let col in data.phase3.missing) {
            html += `<li>${col}: ${data.phase3.missing[col]}</li>`;
        }
        if(Object.keys(data.phase3.missing).length === 0) html += '<li>None</li>';
        html += '</ul>';
        html += `<p><strong>Duplicate Rows:</strong> ${data.phase3.duplicates}</p>`;
        document.getElementById('phase3-content').innerHTML = html;
    }

    // Phase 4
    if(data.phase4) {
        let html = '<div class="code-log">';
        data.phase4.log.forEach(msg => html += `<div>> ${msg}</div>`);
        html += `</div><p><strong>Cleaned Shape:</strong> ${data.phase4.shape.join(' x ')}</p>`;
        document.getElementById('phase4-content').innerHTML = html;
    }

    // Phase 5
    if(data.phase5 && data.phase5.paths) {
        let html = '';
        data.phase5.paths.forEach(p => {
            html += `<img src="${p}?v=${new Date().getTime()}" class="plot-img" alt="EDA Plot">`;
        });
        document.getElementById('phase5-content').innerHTML = html;
    }

    // Phase 6
    if(data.phase6) {
        document.getElementById('phase6-content').innerHTML = `
            <p><strong>Engineered Features:</strong> ${data.phase6.new_features.join(', ')}</p>
            <p><strong>Final Shape before Split:</strong> ${data.phase6.shape.join(' x ')}</p>
        `;
    }

    // Phase 7
    if(data.phase7) {
        document.getElementById('phase7-content').innerHTML = `
            <p><strong>Categorical Features:</strong> ${data.phase7.cat_cols.join(', ')}</p>
            <p><strong>Numerical Features:</strong> ${data.phase7.num_cols.join(', ')}</p>
            <p><strong>Train Shape:</strong> ${data.phase7.train_shape.join(' x ')}</p>
            <p><strong>Test Shape:</strong> ${data.phase7.test_shape.join(' x ')}</p>
        `;
    }

    // Phase 8
    if(data.phase8) {
        let html = '<table><thead><tr><th>Model</th><th>R²</th><th>RMSE</th><th>MAE</th></tr></thead><tbody>';
        data.phase8.forEach(row => {
            html += `<tr>
                <td>${row.Model}</td>
                <td>${row['R²']}</td>
                <td>${row.RMSE}</td>
                <td>${row.MAE}</td>
            </tr>`;
        });
        html += '</tbody></table>';
        document.getElementById('phase8-content').innerHTML = html;
    }

    // Phase 9
    if(data.phase9) {
        let html = '<table><thead><tr><th>Model</th><th>Mean R²</th><th>Std Dev</th></tr></thead><tbody>';
        data.phase9.stats.forEach(row => {
            html += `<tr>
                <td>${row.Model}</td>
                <td>${row['Mean R²']}</td>
                <td>${row['Std Dev']}</td>
            </tr>`;
        });
        html += `</tbody></table>
        <img src="${data.phase9.plot}?v=${new Date().getTime()}" class="plot-img" alt="CV Plot">`;
        document.getElementById('phase9-content').innerHTML = html;
    }

    // Phase 10
    if(data.phase10) {
        let html = `<h4>Best XGBoost Params</h4><ul>`;
        for(let p in data.phase10.best_params) {
            html += `<li>${p}: ${data.phase10.best_params[p]}</li>`;
        }
        html += `</ul>
        <h4>Test Metrics</h4>
        <p>R²: ${data.phase10.metrics['R²']}, RMSE: ${data.phase10.metrics.RMSE}, MAE: ${data.phase10.metrics.MAE}</p>
        <img src="${data.phase10.plot}?v=${new Date().getTime()}" class="plot-img" alt="Tuning Plot">`;
        document.getElementById('phase10-content').innerHTML = html;
    }

    // Phase 11
    if(data.phase11) {
        let html = '<table><thead><tr><th>Feature</th><th>Importance</th></tr></thead><tbody>';
        data.phase11.top_features.forEach(row => {
            html += `<tr><td>${row.Feature}</td><td>${row.Importance.toFixed(4)}</td></tr>`;
        });
        html += `</tbody></table>
        <img src="${data.phase11.plot}?v=${new Date().getTime()}" class="plot-img" alt="Feature Importance">`;
        document.getElementById('phase11-content').innerHTML = html;
    }

    // Phase 12
    if(data.phase12 && data.phase12.paths) {
        let html = '';
        data.phase12.paths.forEach(p => {
            html += `<img src="${p}?v=${new Date().getTime()}" class="plot-img" alt="SHAP Plot">`;
        });
        document.getElementById('phase12-content').innerHTML = html;
    }

    // Phase 13
    if(data.phase13) {
        if(data.phase13.error) {
            document.getElementById('phase13-content').innerHTML = `<p>${data.phase13.error}</p>`;
        } else {
            let html = `
            <h4>Forecast Metrics</h4>
            <p>R²: ${data.phase13.metrics['R²']}, RMSE: ${data.phase13.metrics.RMSE}, MAE: ${data.phase13.metrics.MAE}</p>
            <img src="${data.phase13.plot}?v=${new Date().getTime()}" class="plot-img" alt="Forecast Plot">`;
            document.getElementById('phase13-content').innerHTML = html;
        }
    }

    // Phase 14/15
    if(data.phase15 && data.phase15.paths) {
        let html = '<p>The following models have been saved successfully:</p><ul>';
        data.phase15.paths.forEach(p => {
            html += `<li><code>${p}</code></li>`;
        });
        html += '</ul>';
        document.getElementById('phase15-content').innerHTML = html;
    }
}
