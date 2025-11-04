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

  // Elementos del selector de diccionarios
  const diccionarioSelect = document.getElementById("diccionarioSelect");
  const loadDiccionarioBtn = document.getElementById("loadDiccionarioBtn");
  const diccionarioStatus = document.getElementById("diccionarioStatus");

  let selectedCorpus = null;
  let currentDocuments = [];
  let currentDiccionario = null;
  let lastMetaRes = null; // Guardamos metadatos cargados para mantenerlos visibles

  // ======================
  // CARGAR CORPUS
  // ======================
  fetch("/api/corpora").then(r => r.json()).then(res => {
    if (res.ok) {
      res.data.forEach(c => {
        const li = document.createElement("li");
        li.className = "list-group-item";
        li.innerText = c.nombre;
        li.dataset.id = c.id;
        li.addEventListener("click", function () {
          document.querySelectorAll("#corpusList .list-group-item").forEach(x => x.classList.remove("active"));
          this.classList.add("active");
          selectedCorpus = { id: c.id, nombre: c.nombre };
          selectedCorpusInput.value = c.nombre;
          loadDocuments(c.id);
        });
        corpusListEl.appendChild(li);
      });
    } else {
      corpusListEl.innerHTML = "<li class='list-group-item text-danger'>Error cargando corpora</li>";
    }
  });

  // ======================
  // CARGAR DOCUMENTOS + METADATOS
  // ======================
  function loadDocuments(corpusId) {
    documentsContainer.innerHTML = "<p class='text-muted'>Cargando documentos...</p>";

    // Primero cargamos los metadatos disponibles
    fetch(`/api/metadatos/${corpusId}`).then(r => r.json()).then(metaRes => {
      lastMetaRes = metaRes; // guardamos para reusar después
      renderDocuments(corpusId, metaRes);
    });
  }

  function renderDocuments(corpusId, metaRes, filteredDocs = null) {
    // Panel de metadatos persistente
    let metaPanel = "";
    if (metaRes.ok && metaRes.data.length > 0) {
      metaPanel = `
        <div class="mb-2 p-2 border rounded bg-light">
          <label class="form-label small mb-1">Filtrar por metadato:</label>
          <div class="input-group input-group-sm mb-2">
            <select id="metaSelect" class="form-select">
              <option value="">--Selecciona metadato--</option>
              ${metaRes.data.map(m => `<option value="${m.nombre}">${m.nombre}</option>`).join("")}
            </select>
            <select id="valorSelect" class="form-select">
              <option value="">--Valor--</option>
            </select>
            <button id="applyFilter" class="btn btn-sm btn-primary">Filtrar</button>
          </div>
        </div>`;
    }

    // Si no hay documentos filtrados, los cargamos todos
    if (!filteredDocs) {
      fetch(`/api/documentos/${corpusId}`).then(r => r.json()).then(res => {
        if (res.ok) showDocuments(res.data, corpusId, metaPanel);
        else documentsContainer.innerHTML = `<p class='text-danger'>${res.error}</p>`;
      });
    } else {
      showDocuments(filteredDocs, corpusId, metaPanel);
    }
  }

  function showDocuments(docs, corpusId, metaPanel) {
    currentDocuments = docs;
    if (docs.length === 0) {
      documentsContainer.innerHTML = "<p class='text-muted'>No hay documentos.</p>";
      return;
    }

    const form = document.createElement("div");
    form.innerHTML = metaPanel + `
      <div class='mb-2'>
        <button id='selectAllDocs' class='btn btn-sm btn-outline-secondary'>Seleccionar todo</button>
        <button id='clearAllDocs' class='btn btn-sm btn-outline-secondary'>Limpiar</button>
      </div>`;
    const list = document.createElement("div");

    docs.forEach(doc => {
      const id = doc.id;
      const fila = document.createElement("div");
      fila.className = "form-check";
      fila.innerHTML = `<input class="form-check-input doc-check" type="checkbox" value="${id}" id="doc_${id}">
                        <label class="form-check-label" for="doc_${id}">${doc.archivo}</label>`;
      list.appendChild(fila);
    });

    form.appendChild(list);
    documentsContainer.innerHTML = "";
    documentsContainer.appendChild(form);

    document.getElementById("selectAllDocs").addEventListener("click", () => {
      document.querySelectorAll(".doc-check").forEach(cb => cb.checked = true);
    });
    document.getElementById("clearAllDocs").addEventListener("click", () => {
      document.querySelectorAll(".doc-check").forEach(cb => cb.checked = false);
    });

    const metaSelect = document.getElementById("metaSelect");
    const valorSelect = document.getElementById("valorSelect");
    const applyFilter = document.getElementById("applyFilter");

    if (metaSelect) {
      metaSelect.addEventListener("change", () => {
        const selMeta = lastMetaRes.data.find(m => m.nombre === metaSelect.value);
        valorSelect.innerHTML = "<option value=''>--Valor--</option>";
        if (selMeta) {
          selMeta.valores.forEach(v => {
            valorSelect.innerHTML += `<option value="${v}">${v}</option>`;
          });
        }
      });

      applyFilter.addEventListener("click", () => {
        const meta = metaSelect.value;
        const valor = valorSelect.value;
        if (!meta || !valor) return;
        documentsContainer.innerHTML = "<p class='text-muted'>Aplicando filtro...</p>";
        fetch(`/api/documentos/${corpusId}?meta=${encodeURIComponent(meta)}&valor=${encodeURIComponent(valor)}`)
          .then(r => r.json())
          .then(fres => {
            if (fres.ok) {
              renderDocuments(corpusId, lastMetaRes, fres.data); // Re-render con panel intacto
            } else {
              documentsContainer.innerHTML = `<p class='text-danger'>${fres.error}</p>`;
            }
          });
      });
    }
  }

  // ======================
  // PROCESAR Y GUARDAR DICCIONARIO
  // ======================
  processBtn.addEventListener("click", function () {
    if (!selectedCorpus) {
      alert("Selecciona primero un corpus.");
      return;
    }
    const checked = Array.from(document.querySelectorAll(".doc-check:checked")).map(cb => parseInt(cb.value));
    if (checked.length === 0) {
      alert("Selecciona al menos un documento.");
      return;
    }

    const dicName = prompt("Introduce un nombre para este diccionario:", "NuevoDiccionario");
    if (!dicName) return;

    statusBox.innerText = "Procesando corpus...";
    processBtn.disabled = true;

    fetch("/api/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ corpus_id: selectedCorpus.id, doc_ids: checked, dic_name: dicName })
    })
      .then(r => r.json())
      .then(res => {
        processBtn.disabled = false;
        if (res.ok) {
          statusBox.innerText = " " + res.message;
          const nodes = res.graph.nodes.length;
          const edges = res.graph.edges.length;
          graphSummary.innerHTML = `<strong>Nodos:</strong> ${nodes} — <strong>Aristas:</strong> ${edges}`;
          renderGraphPreview(res.graph);
          currentDiccionario = dicName;
          diccionarioStatus.innerText = `Diccionario activo: ${dicName}`;
          alert("Diccionario guardado correctamente.");
          document.querySelector('#tab2-tab').click();
        } else {
          statusBox.innerText = "Error: " + (res.error || 'error desconocido');
        }
      })
      .catch(err => {
        processBtn.disabled = false;
        statusBox.innerText = "Error en el servidor: " + err;
      });
  });

  // ======================
  // DICCIONARIOS GUARDADOS
  // ======================
  async function loadDiccionarios() {
    const res = await fetch("/api/diccionarios");
    const data = await res.json();
    if (data.ok && data.data.length > 0) {
      diccionarioSelect.innerHTML = data.data.map(d =>
        `<option value="${d.nombre}">${d.nombre}</option>`
      ).join("");
    } else {
      diccionarioSelect.innerHTML = "<option value=''>No hay diccionarios guardados</option>";
    }
  }

  loadDiccionarios();

  loadDiccionarioBtn.addEventListener("click", async () => {
    const nombre = diccionarioSelect.value;
    if (!nombre) {
      alert("Selecciona un diccionario.");
      return;
    }

    diccionarioStatus.innerText = "Cargando diccionario...";
    const res = await fetch("/api/load_diccionario", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ nombre })
    });
    const data = await res.json();
    if (data.ok) {
      diccionarioStatus.innerText = "" + data.message;
      currentDiccionario = nombre;
      const nodes = data.graph.nodes.length;
      const edges = data.graph.edges.length;
      graphSummary.innerHTML = `<strong>Nodos:</strong> ${nodes} — <strong>Aristas:</strong> ${edges}`;
      renderGraphPreview(data.graph);
    } else {
      diccionarioStatus.innerText = "Error: " + data.error;
    }
  });

  // ======================
  // BÚSQUEDA POR DEFINICIÓN
  // ======================
  searchBtn.addEventListener("click", function () {
    const def = definitionInput.value.trim();
    if (!def) { alert("Introduce una definición."); return; }

    if (!currentDiccionario) {
      alert("Selecciona o carga un diccionario antes de buscar.");
      return;
    }

    resultsList.innerHTML = "<li>Cargando...</li>";
    fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ definition: def, top_k: 15, diccionario: currentDiccionario })
    })
      .then(r => r.json())
      .then(res => {
        if (res.ok) {
          resultsList.innerHTML = "";
          res.results.forEach(r => {
            const li = document.createElement("li");
            li.innerText = `${r.palabra}  (score: ${r.score.toFixed(4)})`;
            resultsList.appendChild(li);
          });
        } else {
          resultsList.innerHTML = `<li class='text-danger'>${res.error}</li>`;
        }
      })
      .catch(err => {
        resultsList.innerHTML = `<li class='text-danger'>${err}</li>`;
      });
  });

  // ======================
  // VISTA DE GRAFO
  // ======================
  function renderGraphPreview(graphJson) {
    graphView.innerHTML = "";
    const nNodes = graphJson.nodes.length;
    const nEdges = graphJson.edges.length;
    const summary = document.createElement("div");
    summary.innerHTML = `<p><strong>Nodos (muestra):</strong> ${nNodes}</p><p><strong>Aristas (muestra):</strong> ${nEdges}</p>`;
    graphView.appendChild(summary);

    const ul = document.createElement("ul");
    ul.style.maxHeight = "300px";
    ul.style.overflow = "auto";
    graphJson.nodes.slice(0, 100).forEach(n => {
      const li = document.createElement("li");
      li.innerText = `${n.id} (f:${n.frequency} d:${n.degree})`;
      ul.appendChild(li);
    });
    graphView.appendChild(ul);
  }

  // ======================
  // RECARGAR DICCIONARIOS AL CAMBIAR A PESTAÑA 2
  // ======================
  document.getElementById("tab2-tab").addEventListener("click", function () {
    loadDiccionarios();
  });
});
