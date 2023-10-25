function createTableFromJSON(jsonObj) {
    if (!jsonObj || jsonObj.length === 0) {
        console.error("Invalid or empty JSON object for table creation");
        return null; // or some other error handling
    }

    var table = $('<table></table>');
    var thead = $('<thead></thead>');
    var headerRow = $('<tr></tr>');

    // Explicitly set the order of the keys
    var orderedKeys = ['Method', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'];
    
    orderedKeys.forEach(function(header) {
        headerRow.append('<th>' + header + '</th>');
    });
    
    thead.append(headerRow);
    table.append(thead);
    
    var tbody = $('<tbody></tbody>');
    jsonObj.forEach(function(row) {
        var tableRow = $('<tr></tr>');
        orderedKeys.forEach(function(key) {
            tableRow.append('<td>' + row[key] + '</td>');
        });
        tbody.append(tableRow);
    });
    table.append(tbody);
    
    return table;
}
function setupFormSubmit(endpoint) {
    $('#upload-form').submit(function(e) {
        e.preventDefault();

        $('.results-container').css('display', 'none');
        $('#metrics_id').empty();
        // remove data from previous submission
        $('#psd-plot').attr('src', '');
        $('#raw-data-plot').attr('src', '');
        $('#train-confusion-matrix').attr('src', '');
        $('#test-confusion-matrix').attr('src', '');

        
        var formData = new FormData(this);
        $.ajax({
            url: endpoint,
            type: 'POST',
            data: formData,
            success: function(data) {
    
                $('#psd-plot').attr('src', '');
                $('#raw-data-plot').attr('src', '');
                $('#train-confusion-matrix').attr('src', '');
                $('#test-confusion-matrix').attr('src', '');
                // Step 2: Show the results container when the new data arrives
                $('.results-container').css('display', 'block');

                $('#psd-plot').attr('src', data.psd_path);
                $('#raw-data-plot').attr('src', data.raw_data_path);
                $('#train-confusion-matrix').attr('src', data.confusion_matrix_plot_path);
                
                var table = createTableFromJSON(data.metrics);
                $('#metrics_id').html(table);
            },
            cache: false,
            contentType: false,
            processData: false
        });
    });
}


$(document).ready(function() {
    console.log("jQuery Loaded");
    console.log(window.location.pathname);  
    var endpoint = window.location.pathname === '/train' ? '/train' : '/test';
    setupFormSubmit(endpoint);
});


