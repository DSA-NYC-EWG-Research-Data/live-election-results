import { useEffect, useState, useRef } from 'react'
import { Box, Container, IconButton, Switch, Image } from 'theme-ui'
import { Search, X, ChevronUp, ChevronDown } from 'lucide-react'
import { Themed } from '@theme-ui/mdx'
import maplibregl from 'maplibre-gl'
import {
  Row,
  Column,
  Input,
  Autocomplete,
  Results,
  Options,
} from '../components'
import { addLabels, addShapes } from '../components/layers'
import {
  handleShapeClick,
  handleShapeClickOff,
  handleShapeMouseMove,
  handleShapeMouseMoveoff,
  createShapeMouseMoveHandler,
  createShapeMouseMoveOffHandler,
  createShapeClickHandler,
  createShapeClickOffHandler,
  updateShapeVisibility,
  getMaxKey,
} from '../components/utils'
import { googleProtocol, createGoogleStyle } from 'maplibre-google-maps'
import {
  boroughColors,
  candidates,
  candidateColors,
  mapStyles,
  scaleLookup,
  raceLookup,
} from '../components/constants'
import load from '../components/load'
import shapes from '../data'

const Index = () => {
  const { data, error } = load(
    '/results/Mayoral (FAKE DATA).json',
    '/results/City Council 38 (FAKE DATA).json',
  )

  const map = useRef()
  const [selected, setSelected] = useState({})
  const [selectedCandidate, setSelectedCandidate] = useState('All Candidates')
  const [scale, setScale] = useState('Election district')
  const [race, setRace] = useState('Mayoral')

  const setup = async () => {
    addShapes(map.current, 'election-districts', 0.5)
    addShapes(map.current, 'assembly-districts', 0.5)
    addLabels(map.current)
    const image = await map.current.loadImage('/patterns/cross-hatch.png')
    map.current.addImage('pattern', image.data)
  }

  useEffect(() => {
    map.current = new maplibregl.Map({
      container: 'map',
      style: mapStyles.monochrome,
      center: [-73.956, 40.7228],
      zoom: 10.4,
      minZoom: 10,
    })

    map.current.on('load', () => {
      setup()
    })


    return () => map.current.remove()
  }, [])

  useEffect(() => {
    if (!map.current) return

    const layers = ['election-district', 'assembly-district']
    const handlers = {}

    layers.forEach((layer) => {
      handlers[layer] = {}
      handlers[layer].move = createShapeMouseMoveHandler(map.current, layer)
      handlers[layer].moveOff = createShapeMouseMoveOffHandler(map.current)
      handlers[layer].click = createShapeClickHandler(
        map.current,
        layer,
        setSelected,
      )
      handlers[layer].clickOff = createShapeClickOffHandler(
        map.current,
        layer,
        setSelected,
      )

      const addHandlers = () => {
        map.current.on('mousemove', `${layer}s-fill`, handlers[layer].move)
        map.current.on('click', `${layer}s-fill`, handlers[layer].click)
        map.current.on('mousemove', handlers[layer].moveOff)
        map.current.on('click', handlers[layer].clickOff)
      }

      if (!map.current.isStyleLoaded()) {
        map.current.once('idle', () => {
          addHandlers()
        })
      } else {
        addHandlers()
      }
    })

    return () => {
      layers.forEach((layer) => {
        map.current.off('mousemove', `${layer}s-fill`, handlers[layer].move)
        map.current.off('click', `${layer}s-fill`, handlers[layer].click)
        map.current.off('mousemove', handlers[layer].moveOff)
        map.current.off('click', handlers[layer].clickOff)
      })
    }
  }, [map.current, scale])

  useEffect(() => {
    if (!map.current) return

    const layers = ['election-district', 'assembly-district']

    updateShapeVisibility(map.current, `${scaleLookup[scale]}s`, 'visible')

    layers.forEach((l) => {
      if (!(l == scaleLookup[scale])) {
        updateShapeVisibility(map.current, `${l}s`, 'none')
      }
    })
  }, [map.current, scale])

  useEffect(() => {
    if (!candidates[race].includes(selectedCandidate)) {
      setSelectedCandidate('All Candidates')
    }
  }, [race])

  useEffect(() => {
    const updateSelected = () => {
      if (scale == 'Election district') {
        Object.keys(shapes['election-districts']).forEach((k) => {
          const districtResults = data[raceLookup[race]].election_districts[k]
          let color = '#cccccc'
          let opacity = 0

          if (districtResults) {
            if (selectedCandidate === 'All Candidates') {
              // Show leading candidate for each district
              const leadingCandidate = getMaxKey(districtResults.candidates)
              if (leadingCandidate) {
                color = candidateColors[leadingCandidate] || '#cccccc'
                opacity = districtResults.reporting > 0.1 ? 1 : 0 // Scale opacity based on reporting
              }
            } else if (
              districtResults.candidates[selectedCandidate] !== undefined
            ) {
              // Show only the selected candidate's results
              const totalVotes = Object.values(
                districtResults.candidates,
              ).reduce((a, b) => a + b, 0)
              const candidateVoteShare =
                totalVotes > 0
                  ? districtResults.candidates[selectedCandidate] / totalVotes
                  : 0

              // Show the candidate's color with opacity based on vote share and reporting
              color = candidateColors[selectedCandidate] || '#cccccc'
              opacity = candidateVoteShare
            }
            // If candidate not in district, keep opacity at 0 (invisible)
            map.current.setFeatureState(
              {
                source: 'election-districts',
                id: shapes['election-districts'][k].id,
              },
              {
                color,
                opacity,
                'line-width': selected[scaleLookup[scale]] === k ? 1.5 : 0.5,
              },
            )
          } else {
            map.current.setFeatureState(
              {
                source: 'election-districts',
                id: shapes['election-districts'][k].id,
              },
              {
                color: '#cccccc',
                opacity: 0,
                'line-width': selected[scaleLookup[scale]] === k ? 1.5 : 0.5,
              },
            )
          }
        })
      }

      if (scale == 'Assembly district') {
        Object.keys(shapes['assembly-districts']).forEach((k) => {
          const districtResults = data[raceLookup[race]].assembly_districts[k]
          let color = '#cccccc'
          let opacity = 0

          if (districtResults) {
            if (selectedCandidate === 'All Candidates') {
              // Show leading candidate for each district
              const leadingCandidate = getMaxKey(districtResults.candidates)
              if (leadingCandidate) {
                color = candidateColors[leadingCandidate] || '#cccccc'
                opacity = districtResults.reporting > 0.3 ? 1 : 0 // Scale opacity based on reporting
              }
            } else if (
              districtResults.candidates[selectedCandidate] !== undefined
            ) {
              // Show only the selected candidate's results
              const totalVotes = Object.values(
                districtResults.candidates,
              ).reduce((a, b) => a + b, 0)
              const candidateVoteShare =
                totalVotes > 0
                  ? districtResults.candidates[selectedCandidate] / totalVotes
                  : 0

              // Show the candidate's color with opacity based on vote share and reporting
              color = candidateColors[selectedCandidate] || '#cccccc'
              opacity = candidateVoteShare
            }
            // If candidate not in district, keep opacity at 0 (invisible)
            map.current.setFeatureState(
              {
                source: 'assembly-districts',
                id: shapes['assembly-districts'][k].id,
              },
              {
                color,
                opacity,
                'line-width': selected[scaleLookup[scale]] === k ? 1.5 : 0.5,
              },
            )
          } else {
            map.current.setFeatureState(
              {
                source: 'assembly-districts',
                id: shapes['assembly-districts'][k].id,
              },
              {
                color: '#cccccc',
                opacity: 0,
                'line-width': selected[scaleLookup[scale]] === k ? 1.5 : 0.5,
              },
            )
          }
        })
      }
    }

    if (data[raceLookup[race]]) {
      if (map.current) {
        if (!map.current.isStyleLoaded()) {
          map.current.once('idle', () => {
            updateSelected()
          })
        } else {
          updateSelected()
        }
      }
    }
  }, [data, selected, selectedCandidate, scale, race])

  // const zoomTo = () => {
  //   if (selected) {
  //     const bounds = new maplibregl.LngLatBounds()

  //     let id
  //     zones.features.forEach((feature) => {
  //       if (feature.properties.zone == selected) {
  //         const coords = feature.geometry.coordinates
  //         id = feature.id
  //         if (feature.geometry.type === 'Polygon') {
  //           coords[0].forEach((c) => bounds.extend(c))
  //         } else if (feature.geometry.type === 'MultiPolygon') {
  //           coords.forEach((poly) => {
  //             poly[0].forEach((c) => bounds.extend(c))
  //           })
  //         }
  //       }
  //     })

  //     map.current.fitBounds(bounds, { padding: 150, animate: false })
  //   }
  // }

  const resetView = () => {
    map.current.flyTo({
      center: [-73.956, 40.7228],
      zoom: 10.4,
    })
  }

  return (
    <>
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          zIndex: 1000,
          borderRadius: 2,
        }}
      >
        <Box
          sx={{
            borderRadius: [0, '2px', '2px', '2px'],
            bg: 'rgb(255,255,255,0.9)',
            ml: [0, 4, 4, 4],
            mt: [0, 4, 4, 4],
            width: ['calc(100vw)', '400px', '400px', '400px'],
          }}
        >
          <Results selected={selected} scale={scale} race={race} setSelectedCandidate={setSelectedCandidate}/>
        </Box>
      </Box>
      <Box sx={{ position: 'absolute', top: 0, right: 0 }}>
        <Options
          selectedCandidate={selectedCandidate}
          setSelectedCandidate={setSelectedCandidate}
          scale={scale}
          setScale={setScale}
          race={race}
          setRace={setRace}
        />
      </Box>
      <Box
        sx={{
          position: 'absolute',
          bottom: 0,
          right: 0,
          zIndex: 1000,
          borderRadius: 2,
        }}
      >
        <Image
          sx={{ mb: [-1], mr: [5], width: 200, height: 200 }}
          src='/logos/nycdsa-square-transparent.png'
        />

      </Box>
       <Box
        sx={{
          position: 'absolute',
          bottom: 0,
          let: 0,
          zIndex: 1000,
          borderRadius: 2,
        }}
      >
        <Box sx={{textAlign: 'center', ml: [5], mb: [6], fontSize: [4, 4, 4, 4], fontFamily: 'heading'}}>JOIN DSA</Box>
      </Box>
      <Box
        id='map'
        sx={{
          zIndex: -1,
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100vh',
        }}
      ></Box>
    </>
  )
}

export default Index
