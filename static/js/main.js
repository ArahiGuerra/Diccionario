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
  let lastMetaRes = null; // Guardar metadatos cargados
  let lastFilterSelection = {}; // Guardar selecciones actuales

  // ======================
  // CARGAR CORPUS
  // ======================
  fetch("/api/corpora")
    .then(r => r.json())
    .then(res => {
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
  // CARGAR DOCUMENTOS + METADATOS (con múltiples filtros)
  // ======================
  function loadDocuments(corpusId) {
    documentsContainer.innerHTML = "<p class='text-muted'>Cargando documentos...</p>";

    fetch(`/api/metadatos/${corpusId}`)
      .then(r => r.json())
      .then(metaRes => {
        lastMetaRes = metaRes;
        renderDocuments(corpusId, metaRes);
      });
  }

  function renderDocuments(corpusId, metaRes, filteredDocs = null) {
    // Panel de metadatos persistente
    let metaPanel = "";
    if (metaRes.ok && metaRes.data.length > 0) {
      metaPanel = `
      <div class="mb-2 p-2 border rounded bg-light">
        <label class="form-label small mb-1"> Metadatos disponibles para el corpus:</label>
        <div id="multiMetaPanel" class="mb-2">
          ${metaRes.data
          .map(
            (m, i) => `
            <div class="input-group input-group-sm mb-2">
              <span class="input-group-text">${m.nombre}</span>
              <select id="valor_${i}" class="form-select" data-meta="${m.nombre}">
                <option value="">--Cualquiera--</option>
                ${m.valores
                .map(
                  v =>
                    `<option value="${v}" ${lastFilterSelection[m.nombre] === v ? "selected" : ""
                    }>${v}</option>`
                )
                .join("")}
              </select>
            </div>
          `
          )
          .join("")}
        </div>
      </div>`;
    }

    // Si no hay documentos filtrados, carga todos
    if (!filteredDocs) {
      fetch(`/api/documentos/${corpusId}`)
        .then(r => r.json())
        .then(res => {
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
      documentsContainer.innerHTML = metaPanel + "<p class='text-muted'>No hay documentos.</p>";
      return;
    }

    const form = document.createElement("div");
    form.innerHTML =
      metaPanel +
      `
    <div class='mb-2'>
      <button id='selectAllDocs' class='btn btn-sm btn-outline-secondary'>Seleccionar todo</button>
      <button id='clearAllDocs' class='btn btn-sm btn-outline-secondary'>Limpiar</button>
    </div>`;
    const list = document.createElement("div");

    docs.forEach(doc => {
      const fila = document.createElement("div");
      fila.className = "form-check";
      fila.innerHTML = `<input class="form-check-input doc-check" type="checkbox" value="${doc.id}" id="doc_${doc.id}">
                      <label class="form-check-label" for="doc_${doc.id}">${doc.archivo}</label>`;
      list.appendChild(fila);
    });

    form.appendChild(list);
    documentsContainer.innerHTML = "";
    documentsContainer.appendChild(form);

    document.getElementById("selectAllDocs").addEventListener("click", () => {
      document.querySelectorAll(".doc-check").forEach(cb => (cb.checked = true));
    });
    document.getElementById("clearAllDocs").addEventListener("click", () => {
      document.querySelectorAll(".doc-check").forEach(cb => (cb.checked = false));
    });
  }

  // ======================
  // AUTO-APLICAR FILTRO AL CAMBIAR UN VALOR
  // ======================
  documentsContainer.addEventListener("change", e => {
    if (e.target && e.target.matches("#multiMetaPanel select")) {
      const corpusId = selectedCorpus?.id;
      if (!corpusId) return;

      // Recolectar todos los valores seleccionados
      const selects = documentsContainer.querySelectorAll("#multiMetaPanel select");
      const metas = [];
      const valores = [];
      lastFilterSelection = {};

      selects.forEach(sel => {
        if (sel.value && sel.dataset.meta) {
          metas.push(sel.dataset.meta);
          valores.push(sel.value);
          lastFilterSelection[sel.dataset.meta] = sel.value;
        }
      });

      fetch(
        `/api/documentos/${corpusId}?meta=${encodeURIComponent(metas.join(","))}&valor=${encodeURIComponent(
          valores.join(",")
        )}`
      )
        .then(r => r.json())
        .then(fres => {
          if (fres.ok) {
            renderDocuments(corpusId, lastMetaRes, fres.data);
          } else {
            documentsContainer.innerHTML = `<p class='text-danger'>${fres.error}</p>`;
          }
        });
    }
  });

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
          document.querySelector("#tab2-tab").click();
        } else {
          statusBox.innerText = "Error: " + (res.error || "error desconocido");
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
      diccionarioSelect.innerHTML = data.data.map(d => `<option value="${d.nombre}">${d.nombre}</option>`).join("");
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
    if (!def) {
      alert("Introduce una definición.");
      return;
    }

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
    summary.innerHTML = `<p><strong>Nodos (muestra):</strong> ${nNodes}</p><p><strong>Aristas:</strong> ${nEdges}</p>`;
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
