
document.getElementById('datasetSelect').addEventListener('change', function(event) {
    const selectedDataset = event.target.value;
    if (!selectedDataset || selectedDataset === 'Custom'){
        return;
    }
    // Function to fetch and process JSON data
    function fetchJsonData() {
        $("#loader").show();
        const cellLength = 60;
        fetch('/datasets?id='+ selectedDataset)
        .then(response => response.json())
        .then(data => {
            const textarea = document.getElementById('jsonData');
            textarea.value = JSON.stringify(data, null, 2);
            const keys = data.columns; // Get keys from the first element

            const tableHead = document.getElementById('tableHead');
            const tableBody = document.getElementById('tableBody');
            tableHead.innerHTML = '';
            tableBody.innerHTML = '';
            const headerRow = document.createElement('tr');
            headerRow.innerHTML = `<th></th>`; // Empty cell for checkbox
            keys.forEach(key => {
                const th = document.createElement('th');
                th.textContent = key;
                headerRow.appendChild(th);
            });
            tableHead.appendChild(headerRow);

            data.data.forEach((item, index) => {
                const row = document.createElement('tr');
                // Add checkbox column
                const checkboxCell = document.createElement('td');
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.name = 'selectedRows';
                checkboxCell.appendChild(checkbox);
                row.appendChild(checkboxCell);

                // Add data columns

                keys.forEach(key => {
                    const td = document.createElement('td');
                    const innerDiv = document.createElement('div');
                    innerDiv.classList.add('innerDiv');
                    td.appendChild(innerDiv);
                    const truncatedText = item[key].length > cellLength ? item[key].substring(0, cellLength) + '...' : item[key];
                    innerDiv.title = item[key]; // Set full text as title for hover
                    innerDiv.textContent = item[key];

                  row.appendChild(td);
                });

                tableBody.appendChild(row);
                $("#loader").hide();
            });
        // Show the "Select All" button
        document.getElementById('select-all-container').style.display = 'flex';

        })
        .catch(error => console.error('Error fetching JSON:', error));

    }

        // Call the function when the page loads
        fetchJsonData();


});

document.getElementById('jsonForm').addEventListener('submit', function(event) {
    const selectedIndexes = [];
    const checkboxes = document.querySelectorAll('#tableBody input[type="checkbox"]');
    checkboxes.forEach((checkbox, index) => {
        if (checkbox.checked) {
            selectedIndexes.push(index);
        }
    });
    document.getElementById('selectedIndexes').value = JSON.stringify(selectedIndexes);
});


// Add event listeners for hover effect on dynamically generated cells
document.addEventListener('mouseover', function(event) {
    if (event.target.classList.contains('hoverable')) {
        event.target.style.display = 'fixed '; // Show hover title
        positionHoverTitle(event); // Position hover title
    }
});

function positionHoverTitle(event) {
    console.log(event.target.title);
    const hoverTitle = event.target;
    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight;
    const hoverTitleWidth = hoverTitle.offsetWidth;
    const hoverTitleHeight = hoverTitle.offsetHeight;

    // Calculate position for hover title
    const titleX = (screenWidth - hoverTitleWidth) / 2;
    const titleY = (screenHeight - hoverTitleHeight) / 2;

    // Update hover title position
    hoverTitle.style.left = titleX+ 'px';
    hoverTitle.style.top = titleY + 'px';
}

$("#all-checkbox").change(function(event) {
    $(".method-checkbox").each(function(idx, element) {
        $(this).prop("checked", event.target.checked);
})

});
$("#jsonForm").submit(function(event) {
    $("#loader").show();
    $("#submitButton").prop('disabled', true);

});
