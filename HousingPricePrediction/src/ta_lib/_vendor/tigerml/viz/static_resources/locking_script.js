var target1 = document.querySelector('.plot_data_controls');
var target2 = document.querySelector('.filter_module');
//var target3 = document.querySelector('.splitter_widget');

// create an observer instance
var observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        add_locks();
    });
});

// configuration of the observer:
var config = { attributes: true, childList: true, characterData: true, subtree: true }

// pass in the target node, as well as the observer options
observer.observe(target1, config);
observer.observe(target2, config);
//observer.observe(target3, config);

add_locks();

