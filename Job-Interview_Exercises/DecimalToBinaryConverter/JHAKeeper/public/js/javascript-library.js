
// On interpretation of script element content
// once script element in header of app.blade.php
// has been filled by downloading javascript-library.js...


function processActivityNameIn(inputToUse) {
    makeRowsVisibleBasedOn("activity_name_cell", inputToUse.value);
    lastValueOfInputForActivityName = inputToUse.value;
}


function makeRowsVisibleBasedOn(nameOfTableDataElementToMatch, valueToMatch) {

    var data_rows = document.getElementsByName('presented_jhas_data_row');
    var i;
    var data_cells;
    var j;
    var numberOfDataRowsMadeVisible = 0;
    for (i = 0; i < data_rows.length; i++) {
        data_cells = data_rows[i].getElementsByTagName('td');
        for (j = 0; j < data_cells.length; j++) {
            if ((data_cells[j].getAttribute('name') == nameOfTableDataElementToMatch) &&
                (data_cells[j].innerHTML == valueToMatch)) {
                data_rows[i].style.visibility = 'visible';
                numberOfDataRowsMadeVisible += 1;
                break;
            }
            else {
                data_rows[i].style.visibility = 'collapse';
            }
        }
    }

    var header_row = document.getElementById('presented_jhas_header_row');
    if (numberOfDataRowsMadeVisible > 0) {
        header_row.style.visibility = 'visible';
    }
    else {
        header_row.style.visibility = 'collapse';
    }
}
