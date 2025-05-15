const projects = [
    {
        title: "Example Project 1",
        desc: "Short description for Example Project 1.",
        readme: "Projects/ExampleProject1/README.md",
        image: "Projects/ExampleProject1/screenshot.png"
    },
    {
        title: "Example Project 2",
        desc: "Short description for Example Project 2.",
        readme: "Projects/ExampleProject2/README.md",
        image: "Projects/ExampleProject2/screenshot.png"
    }
];

function renderProjects() {
    const container = document.getElementById('project-cards');
    container.innerHTML = projects.map(proj => `
        <div class="card">
            ${proj.image ? `<img src="${proj.image}" alt="${proj.title} screenshot">` : ''}
            <div class="card-title">${proj.title}</div>
            <div class="card-desc">${proj.desc}</div>
            <div class="card-links">
                <a href="${proj.readme}" target="_blank">README</a>
            </div>
        </div>
    `).join('');
}
renderProjects();

document.getElementById('toggle-dark').onclick = function () {
    document.body.classList.toggle('dark');
};
