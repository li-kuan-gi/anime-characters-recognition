fetch("./model/characters.json")
    .then((r) => r.json())
    .then((json) => {
        window.characters = json;
        json.forEach(char => createCharacterInfo(char));
    });

function createCharacterInfo(char) {
    const table = document.getElementById('support');

    const row = document.createElement('tr');

    const work = document.createElement('td');
    work.innerHTML = char.work;

    const name = document.createElement('td');
    name.innerHTML = char.name;

    row.appendChild(work);
    row.appendChild(name);

    table.appendChild(row);
}