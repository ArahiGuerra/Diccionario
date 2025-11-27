document.addEventListener("DOMContentLoaded", function () {
  const corpusListEl = document.getElementById("corpusList");
  const documentsContainer = document.getElementById("documentsContainer");
  const selectedCorpusInput = document.getElementById("selectedCorpus");
  const processBtn = document.getElementById("processBtn");
  const statusBox = document.getElementById("statusBox");
  const graphSummary = document.getElementById("graphSummary");
  const graphView = document.getElementById("graphView");
  const definitionInput = document.getElementById("definitionInput");
  const searchBtn = document.getElementById("searchBtn");
  const resultsList = document.getElementById("resultsList");

  // Prefijo URL (para hosting en subcarpeta)
  let locationPathName = location.pathname;
  if (locationPathName === "/") locationPathName = "";

  const diccionarioSelect = document.getElementById("diccionarioSelect");
  const loadDiccionarioBtn = document.getElementById("loadDiccionarioBtn");
  const diccionarioStatus = document.getElementById("diccionarioStatus");

  let selectedCorpus = null;
  let currentDocuments = [];
  let currentDiccionario = null;

  let lastMetaRes = null; // metadatos cargados
  let lastFilterSelection = {}; // { "Área": "Medicina", "Tipo": "Wikipedia" }

  //  Colección GLOBAL de documentos seleccionados (se conserva entre filtros)
  let globalSelectedDocs = new Set();

  // ======================
  // CARGAR CORPUS
  // ======================
  fetch(locationPathName + "/api/corpora")
    .then(r => r.json())
    .then(res => {
      if (res.ok) {
        res.data.forEach(c => {
          const li = document.createElement("li");
          li.className = "list-group-item";
          li.innerText = c.nombre;
          li.dataset.id = c.id;

          li.addEventListener("click", function () {
            document.querySelectorAll("#corpusList .list-group-item").forEach(x =>
              x.classList.remove("active")
            );
            this.classList.add("active");

            selectedCorpus = { id: c.id, nombre: c.nombre };
            selectedCorpusInput.value = c.nombre;

            // Reiniciamos selección global al cambiar de corpus
            globalSelectedDocs.clear();
            lastFilterSelection = {};

            loadDocuments(c.id);
          });

          corpusListEl.appendChild(li);
        });
      } else {
        corpusListEl.innerHTML =
          "<li class='list-group-item text-danger'>Error cargando corpora</li>";
      }
    });

  // ======================
  // CARGAR DOCUMENTOS + METADATOS
  // ======================
  function loadDocuments(corpusId) {
    documentsContainer.innerHTML =
      "<p class='text-muted'>Cargando documentos...</p>";

    fetch(locationPathName + `/api/metadatos/${corpusId}`)
      .then(r => r.json())
      .then(metaRes => {
        lastMetaRes = metaRes;
        renderDocuments(corpusId, metaRes);
      });
  }

  function renderDocuments(corpusId, metaRes, filteredDocs = null) {
    let metaPanel = "";

    if (metaRes.ok) {
      metaPanel = `
      <div class="mb-2 p-2 border rounded bg-light">
        <label class="form-label small mb-1"><b>Filtrar por metadatos:</b></label>

        <div id="multiMetaPanel" class="mb-2">
          ${metaRes.data
            .map((m, i) => {
              const selectedVal = lastFilterSelection[m.nombre] || "";
              return `
            <div class="input-group input-group-sm mb-2">
              <span class="input-group-text">${m.nombre}</span>

              <select id="valor_${i}" class="form-select" data-meta="${m.nombre}">
                <option value="">--Cualquiera--</option>
                ${m.valores
                  .map(
                    v =>
                      `<option value="${v}" ${
                        selectedVal === v ? "selected" : ""
                      }>${v}</option>`
                  )
                  .join("")}
              </select>
            </div>`;
            })
            .join("")}
        </div>
      </div>`;
    }

    if (!filteredDocs) {
      fetch(locationPathName + `/api/documentos/${corpusId}`)
        .then(r => r.json())
        .then(res => {
          if (res.ok) showDocuments(res.data, corpusId, metaPanel);
          else
            documentsContainer.innerHTML = `<p class="text-danger">${res.error}</p>`;
        });
    } else {
      showDocuments(filteredDocs, corpusId, metaPanel);
    }
  }

  // ======================
  // MOSTRAR DOCUMENTOS
  // ======================
  function showDocuments(docs, corpusId, metaPanel) {
    currentDocuments = docs;

    let html = metaPanel;

    //  Sección de documentos seleccionados
    if (globalSelectedDocs.size > 0) {
      // Intentar obtener nombres de todos los documentos seleccionados
      fetch(locationPathName + `/api/documentos/${corpusId}`)
        .then(r => r.json())
        .then(allDocsRes => {
          if (allDocsRes.ok) {
            const selectedDocsInfo = Array.from(globalSelectedDocs)
              .map(id => {
                const doc = allDocsRes.data.find(d => d.id === id);
                return doc ? doc.archivo : `ID: ${id}`;
              })
              .join(", ");
            
            const alertDiv = document.querySelector(".selected-docs-alert");
            if (alertDiv) {
              alertDiv.innerHTML = `
                <strong> Documentos seleccionados (${globalSelectedDocs.size}):</strong>
                <div class='small mt-1' style='max-height: 100px; overflow-y: auto;'>
                  ${selectedDocsInfo}
                </div>`;
            }
          }
        });

      html += `
      <div class='alert alert-info mb-3 selected-docs-alert'>
        <strong> Documentos seleccionados (${globalSelectedDocs.size}):</strong>
        <div class='small mt-1'>Cargando nombres...</div>
      </div>`;
    }

    if (docs.length === 0) {
      documentsContainer.innerHTML =
        metaPanel + "<p class='text-muted'>No hay documentos.</p>";
      return;
    }

    html += `
    <div class='mb-2'>
      <button id='selectAllDocs' class='btn btn-sm btn-outline-secondary'>Seleccionar visibles</button>
      <button id='clearAllDocs' class='btn btn-sm btn-outline-secondary'>Deseleccionar visibles</button>
    </div>`;

    const container = document.createElement("div");
    container.innerHTML = html;

    const list = document.createElement("div");

    docs.forEach(doc => {
      const fila = document.createElement("div");
      fila.className = "form-check";

      const checked = globalSelectedDocs.has(doc.id) ? "checked" : "";

      fila.innerHTML = `
      <input class="form-check-input doc-check" type="checkbox" value="${doc.id}" id="doc_${doc.id}" ${checked}>
      <label class="form-check-label" for="doc_${doc.id}">${doc.archivo}</label>
    `;

      list.appendChild(fila);
    });

    container.appendChild(list);
    documentsContainer.innerHTML = "";
    documentsContainer.appendChild(container);

    // Marcar / desmarcar visibles
    document.getElementById("selectAllDocs").addEventListener("click", () => {
      document.querySelectorAll(".doc-check").forEach(cb => {
        cb.checked = true;
        globalSelectedDocs.add(parseInt(cb.value));
      });
      // Actualizar el contador
      loadDocuments(corpusId);
    });

    document.getElementById("clearAllDocs").addEventListener("click", () => {
      document.querySelectorAll(".doc-check").forEach(cb => {
        cb.checked = false;
        globalSelectedDocs.delete(parseInt(cb.value));
      });
      // Actualizar el contador
      loadDocuments(corpusId);
    });

    // Mantener selección global
    document
      .querySelectorAll(".doc-check")
      .forEach(cb => cb.addEventListener("change", e => {
        const id = parseInt(cb.value);
        if (cb.checked) globalSelectedDocs.add(id);
        else globalSelectedDocs.delete(id);
        
        // Actualizar la visualización del contador
        const alertDiv = document.querySelector(".selected-docs-alert");
        if (alertDiv && globalSelectedDocs.size > 0) {
          fetch(locationPathName + `/api/documentos/${corpusId}`)
            .then(r => r.json())
            .then(allDocsRes => {
              if (allDocsRes.ok) {
                const selectedDocsInfo = Array.from(globalSelectedDocs)
                  .map(id => {
                    const doc = allDocsRes.data.find(d => d.id === id);
                    return doc ? doc.archivo : `ID: ${id}`;
                  })
                  .join(", ");
                
                alertDiv.innerHTML = `
                  <strong> Documentos seleccionados (${globalSelectedDocs.size}):</strong>
                  <div class='small mt-1' style='max-height: 100px; overflow-y: auto;'>
                    ${selectedDocsInfo}
                  </div>`;
              }
            });
        } else if (alertDiv && globalSelectedDocs.size === 0) {
          alertDiv.remove();
        }
      }));
  }

  // =====================================================
  // AUTO-APLICAR FILTROS AL CAMBIAR METADATOS
  // =====================================================
  documentsContainer.addEventListener("change", e => {
    if (!e.target.matches("#multiMetaPanel select")) return;

    const corpusId = selectedCorpus?.id;
    if (!corpusId) return;

    const selects = document.querySelectorAll("#multiMetaPanel select");

    const metas = [];
    const valores = [];

    lastFilterSelection = {};

    selects.forEach(sel => {
      if (sel.value) {
        metas.push(sel.dataset.meta);
        valores.push(sel.value);
        lastFilterSelection[sel.dataset.meta] = sel.value;
      }
    });

    if (metas.length === 0) {
      loadDocuments(corpusId);
      return;
    }

    documentsContainer.innerHTML =
      "<p class='text-muted'>Aplicando filtros...</p>";

    fetch(
      locationPathName +
        `/api/documentos/${corpusId}?meta=${encodeURIComponent(
          metas.join(",")
        )}&valor=${encodeURIComponent(valores.join(","))}`
    )
      .then(r => r.json())
      .then(fres => {
        if (fres.ok) {
          renderDocuments(corpusId, lastMetaRes, fres.data);
        } else {
          documentsContainer.innerHTML =
            `<p class='text-danger'>${fres.error}</p>`;
        }
      });
  });

  // ======================
  // PROCESAR DICCIONARIO
  // ======================
  processBtn.addEventListener("click", function () {
    if (!selectedCorpus) {
      alert("Selecciona primero un corpus.");
      return;
    }

    if (globalSelectedDocs.size === 0) {
      alert("Selecciona al menos un documento.");
      return;
    }

    const checked = Array.from(globalSelectedDocs);

    const dicName = prompt(
      "Introduce un nombre para este diccionario:",
      "NuevoDiccionario"
    );
    if (!dicName) return;

    statusBox.innerText = "Procesando corpus...";
    processBtn.disabled = true;

    fetch(locationPathName + "/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        corpus_id: selectedCorpus.id,
        doc_ids: checked,
        dic_name: dicName
      })
    })
      .then(r => r.json())
      .then(res => {
        processBtn.disabled = false;

        if (res.ok) {
          statusBox.innerText = res.message;
          const nodes = res.graph.nodes.length;
          const edges = res.graph.edges.length;

          graphSummary.innerHTML = `<strong>Nodos:</strong> ${nodes} — <strong>Aristas:</strong> ${edges}`;
          renderGraphPreview(res.graph);

          currentDiccionario = dicName;
          diccionarioStatus.innerText =
            `Diccionario activo: ${dicName}`;

          alert("Diccionario guardado correctamente.");
          document.querySelector("#tab2-tab").click();
        } else {
          statusBox.innerText = "Error: " + res.error;
        }
      })
      .catch(err => {
        processBtn.disabled = false;
        statusBox.innerText = "Error en el servidor: " + err;
      });
  });

  // ======================
  // RESTO DEL CÓDIGO (diccionarios, búsqueda, vista de grafo)
  // ======================

  async function loadDiccionarios() {
    const res = await fetch(locationPathName + "/api/diccionarios");
    const data = await res.json();

    if (data.ok && data.data.length > 0) {
      diccionarioSelect.innerHTML = data.data
        .map(d => `<option value="${d.nombre}">${d.nombre}</option>`)
        .join("");
    } else {
      diccionarioSelect.innerHTML =
        "<option value=''>No hay diccionarios guardados</option>";
    }
  }

  loadDiccionarios();

  loadDiccionarioBtn.addEventListener("click", async () => {
    const nombre = diccionarioSelect.value;
    if (!nombre) {
      alert("Selecciona un diccionario.");
      return;
    }

    diccionarioStatus.innerText = "Cargando...";

    const res = await fetch(locationPathName + "/api/load_diccionario", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nombre })
    });

    const data = await res.json();

    if (data.ok) {
      diccionarioStatus.innerText = data.message;

      currentDiccionario = nombre;
      const nodes = data.graph.nodes.length;
      const edges = data.graph.edges.length;

      graphSummary.innerHTML = `<strong>Nodos:</strong> ${nodes} — <strong>Aristas:</strong> ${edges}`;
      renderGraphPreview(data.graph);
    } else {
      diccionarioStatus.innerText = "Error: " + data.error;
    }
  });

  searchBtn.addEventListener("click", function () {
    const def = definitionInput.value.trim();
    if (!def) {
      alert("Introduce una definición.");
      return;
    }

    if (!currentDiccionario) {
      alert("Selecciona o carga un diccionario.");
      return;
    }

    resultsList.innerHTML = "<li>Cargando...</li>";

    fetch(locationPathName + "/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        definition: def,
        top_k: 15,
        diccionario: currentDiccionario
      })
    })
      .then(r => r.json())
      .then(res => {
        if (res.ok) {
          resultsList.innerHTML = "";
          res.results.forEach(r => {
            const li = document.createElement("li");
            li.innerText =
              `${r.palabra} (score: ${r.score.toFixed(4)})`;
            resultsList.appendChild(li);
          });
        } else {
          resultsList.innerHTML =
            `<li class="text-danger">${res.error}</li>`;
        }
      });
  });

  function renderGraphPreview(graphJson) {
    graphView.innerHTML = "";

    const summary = document.createElement("div");
    summary.innerHTML = `
      <p><strong>Nodos (muestra):</strong> ${graphJson.nodes.length}</p>
      <p><strong>Aristas (muestra):</strong> ${graphJson.edges.length}</p>`;
    graphView.appendChild(summary);

    const ul = document.createElement("ul");
    ul.style.maxHeight = "300px";
    ul.style.overflow = "auto";

    graphJson.nodes.slice(0, 100).forEach(n => {
      const li = document.createElement("li");
      li.innerText =
        `${n.id} (f:${n.frequency}, d:${n.degree})`;
      ul.appendChild(li);
    });

    graphView.appendChild(ul);
  }

  document.getElementById("tab2-tab").addEventListener("click", function () {
    loadDiccionarios();
  });
});